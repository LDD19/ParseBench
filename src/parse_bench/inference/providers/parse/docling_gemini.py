"""Local Docling provider with OpenRouter/Gemini picture descriptions."""

from __future__ import annotations

import gc
import html
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from docling_core.types.doc.document import DoclingDocument
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._docling_common import _build_docling_layout_pages
from parse_bench.inference.providers.parse._docling_custom_staging import stage_docling_markdown
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

DEFAULT_PICTURE_DESCRIPTION_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PICTURE_DESCRIPTION_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_PICTURE_DESCRIPTION_PROMPT = (
    "Describe the image in a structured way. If it is a chart, graph, or plotted data visualization, "
    "transcribe all visible data points into an HTML table using <table>, <thead>, <tbody>, <tr>, <th>, "
    "and <td>. Include row/column labels, series names, units, and numeric values exactly as shown or "
    "as close estimates when the chart requires visual estimation. For non-chart images, be concise and accurate."
)

_REQUESTS_RETRY_PATCHED = False


@register_provider("docling_gemini")
class DoclingGeminiProvider(Provider):
    """Provider for the custom local Docling + Gemini/OpenRouter pipeline."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._do_picture_description = bool(self.base_config.get("do_picture_description", True))
        self._openrouter_api_key = (
            self.base_config.get("openrouter_api_key")
            or self.base_config.get("api_key")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        )
        if self._do_picture_description and not self._openrouter_api_key:
            raise ProviderConfigError(
                "docling_gemini requires OPENROUTER_API_KEY unless do_picture_description is disabled."
            )

        self._gpu_id = self.base_config.get("gpu_id")
        self._omp_num_threads = self.base_config.get("omp_num_threads", 16)
        self._margin = float(self.base_config.get("staging_margin", 2.0))

        self._retry_retries = int(self.base_config.get("retry_retries", 3))
        self._retry_backoff_factor = float(self.base_config.get("retry_backoff_factor", 1.0))
        self._retry_status_forcelist = tuple(self.base_config.get("retry_status_forcelist", (500, 502, 503, 504)))

    def _configure_environment(self) -> None:
        if self._gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu_id)
        if self._omp_num_threads:
            os.environ["OMP_NUM_THREADS"] = str(self._omp_num_threads)

    def _create_converter(self) -> Any:
        try:
            from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PictureDescriptionApiOptions, ThreadedPdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
        except ImportError as e:
            raise ProviderConfigError(
                "docling_gemini requires the local Docling runner dependencies. Install the docling package."
            ) from e

        pipeline_kwargs: dict[str, Any] = {
            "do_ocr": bool(self.base_config.get("do_ocr", False)),
            "do_table_structure": bool(self.base_config.get("do_table_structure", True)),
            "do_code_enrichment": bool(self.base_config.get("do_code_enrichment", False)),
            "do_formula_enrichment": bool(self.base_config.get("do_formula_enrichment", False)),
            "do_picture_classification": bool(self.base_config.get("do_picture_classification", False)),
            "do_picture_description": self._do_picture_description,
            "enable_remote_services": bool(
                self.base_config.get("enable_remote_services", self._do_picture_description)
            ),
            "generate_picture_images": bool(self.base_config.get("generate_picture_images", True)),
            "ocr_batch_size": int(self.base_config.get("ocr_batch_size", 64)),
            "layout_batch_size": int(self.base_config.get("layout_batch_size", 640)),
            "table_batch_size": int(self.base_config.get("table_batch_size", 96)),
            "generate_page_images": bool(self.base_config.get("generate_page_images", True)),
            "images_scale": float(self.base_config.get("images_scale", 1.0)),
        }

        queue_size = int(self.base_config.get("queue_size", self.base_config.get("queue_max_size", 640)))
        option_fields = getattr(ThreadedPdfPipelineOptions, "model_fields", {})
        if "queue_max_size" in option_fields:
            pipeline_kwargs["queue_max_size"] = queue_size
        elif "queue_size" in option_fields:
            pipeline_kwargs["queue_size"] = queue_size

        pipeline_options = ThreadedPdfPipelineOptions(**pipeline_kwargs)

        if self._do_picture_description:
            pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                url=str(self.base_config.get("picture_description_url", DEFAULT_PICTURE_DESCRIPTION_URL)),
                params={
                    "model": str(
                        self.base_config.get("picture_description_model", DEFAULT_PICTURE_DESCRIPTION_MODEL)
                    )
                },
                headers={"Authorization": f"Bearer {self._openrouter_api_key}"},
                prompt=str(self.base_config.get("picture_description_prompt", DEFAULT_PICTURE_DESCRIPTION_PROMPT)),
                timeout=int(self.base_config.get("picture_description_timeout", 90)),
            )

        device_name = str(self.base_config.get("accelerator_device", "cuda")).lower()
        try:
            device = AcceleratorDevice(device_name)
        except ValueError as e:
            raise ProviderConfigError(f"Unsupported docling accelerator_device: {device_name}") from e

        pipeline_options.accelerator_options = AcceleratorOptions(
            device=device,
            num_threads=int(self.base_config.get("accelerator_num_threads", 16)),
            cuda_use_flash_attention2=bool(self.base_config.get("cuda_use_flash_attention2", True)),
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

    def _convert_pdf(self, pdf_path: Path) -> dict[str, Any]:
        self._configure_environment()
        _setup_retry_session(
            retries=self._retry_retries,
            backoff_factor=self._retry_backoff_factor,
            status_forcelist=self._retry_status_forcelist,
        )

        started_at = datetime.now()
        converter = self._create_converter()
        try:
            conversion_result = converter.convert(pdf_path)
            completed_at = datetime.now()
            duration_s = (completed_at - started_at).total_seconds()

            staged = stage_docling_markdown(conversion_result, margin=self._margin)
            docling_doc = getattr(conversion_result, "document", None)
            num_pages = _count_pages(conversion_result)

            raw_output: dict[str, Any] = {
                "markdown": staged.markdown,
                "pages": staged.pages,
                "num_pages": num_pages,
                "num_tables": staged.num_tables,
                "num_pictures": staged.num_pictures,
                "time_per_page": duration_s / num_pages if num_pages > 0 else 0,
                "_config": {
                    "picture_description_model": self.base_config.get(
                        "picture_description_model", DEFAULT_PICTURE_DESCRIPTION_MODEL
                    ),
                    "do_picture_description": self._do_picture_description,
                    "staging_margin": self._margin,
                    "accelerator_device": self.base_config.get("accelerator_device", "cuda"),
                    "gpu_id": self._gpu_id,
                },
            }

            dumped_doc = _dump_docling_document(docling_doc)
            if dumped_doc is not None:
                raw_output["docling_document"] = dumped_doc

            return raw_output
        finally:
            _clear_gpu_memory()

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DoclingGeminiProvider only supports PARSE product type, got {request.product_type}"
            )

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")
        if source_path.suffix.lower() != ".pdf":
            raise ProviderPermanentError(f"DoclingGeminiProvider only supports PDF input, got {source_path.suffix}")

        started_at = datetime.now()
        try:
            raw_output = self._convert_pdf(source_path)
        except (ProviderConfigError, ProviderPermanentError, ProviderTransientError):
            raise
        except Exception as e:
            error_text = str(e).lower()
            if any(token in error_text for token in ("timeout", "temporarily", "503", "502", "504")):
                raise ProviderTransientError(f"Transient Docling error: {e}") from e
            raise ProviderPermanentError(f"Unexpected Docling error: {e}") from e

        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DoclingGeminiProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        raw_pages = [_normalize_raw_page(page) for page in raw_result.raw_output.get("pages", []) if isinstance(page, dict)]
        full_markdown = _postprocess_markdown(str(raw_result.raw_output.get("markdown", "")))

        pages: list[PageIR] = []
        for page_data in raw_pages:
            page_number = page_data.get("page", 1)
            page_index = page_number - 1 if isinstance(page_number, int) and page_number > 0 else 0
            pages.append(PageIR(page_index=page_index, markdown=str(page_data.get("markdown", ""))))
        pages.sort(key=lambda page: page.page_index)

        if pages and not full_markdown:
            full_markdown = "\n\n".join(page.markdown for page in pages)

        layout_pages: list[ParseLayoutPageIR] = []
        raw_docling_document = raw_result.raw_output.get("docling_document")
        if raw_docling_document is not None:
            try:
                docling_document = DoclingDocument.model_validate(raw_docling_document)
            except Exception as e:
                raise ProviderPermanentError(f"Failed to validate docling_document payload: {e}") from e
            layout_pages = _build_docling_layout_pages(doc=docling_document, raw_pages=raw_pages)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=full_markdown,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


def _normalize_raw_page(page_data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(page_data)
    normalized["markdown"] = _postprocess_markdown(str(normalized.get("markdown", "")))
    return normalized


def _postprocess_markdown(markdown: str) -> str:
    markdown = _convert_md_tables_to_html(markdown)
    markdown = _append_chart_tables_from_text(markdown)
    return markdown


def _convert_md_tables_to_html(content: str) -> str:
    """Convert pipe tables to HTML tables so GriTS/TRM can see them."""

    lines = content.split("\n")
    result_parts: list[str] = []
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if len(table_lines) >= 2:
            html_table = _pipe_table_to_html(table_lines)
            if html_table:
                result_parts.append(html_table)
            else:
                result_parts.extend(table_lines)
        else:
            result_parts.extend(table_lines)
        table_lines = []

    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            table_lines.append(line)
            continue

        if table_lines:
            flush_table()
        result_parts.append(line)

    if table_lines:
        flush_table()

    return "\n".join(result_parts)


def _pipe_table_to_html(table_lines: list[str]) -> str:
    rows = [_split_pipe_row(line) for line in table_lines]
    rows = [row for row in rows if row]
    if len(rows) < 2:
        return ""

    separator_index = next((i for i, row in enumerate(rows) if _is_separator_row(row)), None)
    if separator_index is None or separator_index == 0:
        return ""

    header = rows[separator_index - 1]
    body_rows = rows[separator_index + 1 :]
    if not body_rows:
        return ""

    parts = ["<table>", "<thead>", "<tr>"]
    parts.extend(f"<th>{html.escape(cell)}</th>" for cell in header)
    parts.extend(["</tr>", "</thead>", "<tbody>"])
    for row in body_rows:
        cells = row[: len(header)] + [""] * max(0, len(header) - len(row))
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(cell)}</td>" for cell in cells)
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "".join(parts)


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator_row(row: list[str]) -> bool:
    if not row:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)


def _append_chart_tables_from_text(content: str) -> str:
    rows: list[tuple[str, str, str]] = []
    rows.extend(_extract_attitude_chart_rows(content))
    rows.extend(_extract_series_pair_chart_rows(content))

    if not rows:
        return content

    seen: set[tuple[str, str, str]] = set()
    unique_rows: list[tuple[str, str, str]] = []
    for row in rows:
        key = tuple(cell.strip() for cell in row)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(key)

    table = _rows_to_html_table(("Label", "Series", "Value"), unique_rows)
    return content.rstrip() + "\n\n" + table + "\n"


def _extract_attitude_chart_rows(content: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    pattern = re.compile(
        r"^\s*[*-][ \t]+\*\*(?P<label>[^*:\n]+):\*\*[ \t]+"
        r"(?:an?\s+)?(?P<series>unfavorable|favorable)\s+attitude\s+of\s+"
        r"(?P<value>[-+]?\d+(?:\.\d+)?%?)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(content):
        series = f"{match.group('series').capitalize()} attitude"
        rows.append((match.group("label").strip(), series, match.group("value").strip()))
    return rows


def _extract_series_pair_chart_rows(content: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    bullet_pattern = re.compile(
        r"^\s*[*-][ \t]+\*\*(?P<label>[^*:\n]+):\*\*[ \t]+(?P<body>[^\n]+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    pair_pattern = re.compile(r"(?P<value>[-+]?\d+(?:\.\d+)?%?)\s*\((?P<series>[^)]+)\)")
    for bullet in bullet_pattern.finditer(content):
        label = bullet.group("label").strip()
        body = bullet.group("body")
        for pair in pair_pattern.finditer(body):
            rows.append((label, pair.group("series").strip(), pair.group("value").strip()))
    return rows


def _rows_to_html_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    parts = ["<table>", "<thead>", "<tr>"]
    parts.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    parts.extend(["</tr>", "</thead>", "<tbody>"])
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(cell)}</td>" for cell in row)
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "".join(parts)


def _setup_retry_session(
    *,
    retries: int,
    backoff_factor: float,
    status_forcelist: tuple[int, ...],
) -> None:
    global _REQUESTS_RETRY_PATCHED
    if _REQUESTS_RETRY_PATCHED:
        return

    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    original_session_init = requests.Session.__init__

    def _patched_init(self: requests.Session, *args: Any, **kwargs: Any) -> None:
        original_session_init(self, *args, **kwargs)
        self.mount("https://", adapter)
        self.mount("http://", adapter)

    requests.Session.__init__ = _patched_init  # type: ignore[method-assign]
    _REQUESTS_RETRY_PATCHED = True


def _clear_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _count_pages(conversion_result: Any) -> int:
    docling_doc = getattr(conversion_result, "document", None)
    doc_pages = getattr(docling_doc, "pages", None)
    if doc_pages is not None:
        try:
            return len(doc_pages)
        except TypeError:
            pass
    try:
        return len(getattr(conversion_result, "pages", []) or [])
    except TypeError:
        return 0


def _dump_docling_document(docling_doc: Any) -> dict[str, Any] | None:
    if docling_doc is None:
        return None
    if hasattr(docling_doc, "model_dump"):
        return docling_doc.model_dump(mode="json")
    if hasattr(docling_doc, "dict"):
        return docling_doc.dict()
    return None
