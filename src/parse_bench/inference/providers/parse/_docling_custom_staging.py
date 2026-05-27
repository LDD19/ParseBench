"""Custom Docling staging/export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DocItemLabel, TableData, TableItem


@dataclass(frozen=True)
class StagedDoclingMarkdown:
    markdown: str
    pages: list[dict[str, Any]]
    num_tables: int
    num_pictures: int


def stage_docling_markdown(conversion_result: Any, *, margin: float = 2.0) -> StagedDoclingMarkdown:
    """Apply the custom header relabel pass and export staged markdown."""

    doc_pages = list(getattr(conversion_result, "pages", []) or [])
    docling_doc = getattr(conversion_result, "document", None)

    _relabel_repeated_headers(doc_pages, margin=margin)

    doc_metadata = _build_doc_metadata(docling_doc)
    picture_descriptions = _extract_picture_descriptions(docling_doc)

    full_parts: list[str] = []
    raw_pages: list[dict[str, Any]] = []
    counter_tables = 0
    counter_pictures = 0

    for page_index, page in enumerate(doc_pages):
        if not getattr(page, "assembled", None) or not getattr(page.assembled, "elements", None):
            continue

        page_parts: list[str] = [f"<page_number>{page_index + 1}</page_number>\n\n\n"]
        page_header_markdown = _collect_page_section_markdown(page.assembled.elements, DocItemLabel.PAGE_HEADER)
        page_footer_markdown = _collect_page_section_markdown(page.assembled.elements, DocItemLabel.PAGE_FOOTER)
        previous_label = None

        for element in page.assembled.elements:
            if previous_label == DocItemLabel.LIST_ITEM and element.label != DocItemLabel.LIST_ITEM:
                page_parts.append("\n\n")

            if element.label == DocItemLabel.SECTION_HEADER:
                page_parts.append(f"## {getattr(element, 'text', '')}\n")

            elif element.label in {DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX}:
                html_table = _export_element_table(
                    element=element,
                    doc_metadata=doc_metadata,
                    table_index=counter_tables,
                )
                if html_table:
                    page_parts.append(html_table)
                    page_parts.append("\n\n\n")
                counter_tables += 1

            elif element.label == DocItemLabel.PICTURE:
                cref = f"#/pictures/{counter_pictures}"
                page_parts.append(f"<!-- image ref: {cref} -->\n")
                description = (
                    picture_descriptions[counter_pictures]
                    if counter_pictures < len(picture_descriptions)
                    else None
                )
                if description and description.strip():
                    page_parts.append(f"{description}\n")
                else:
                    page_parts.append("<!-- No description available -->\n")
                page_parts.append("\n\n")
                counter_pictures += 1

            elif element.label == DocItemLabel.KEY_VALUE_REGION:
                cluster = getattr(element, "cluster", None)
                cells = getattr(cluster, "cells", []) if cluster is not None else []
                for cell in cells:
                    page_parts.append(f"{getattr(cell, 'text', '')}\n")
                page_parts.append("\n\n")

            elif element.label in {DocItemLabel.CAPTION, DocItemLabel.TEXT, DocItemLabel.FOOTNOTE}:
                page_parts.append(f"{getattr(element, 'text', '')}\n\n\n")

            elif element.label == DocItemLabel.LIST_ITEM:
                page_parts.append(f"{getattr(element, 'text', '')}\n")

            previous_label = element.label

        if previous_label == DocItemLabel.LIST_ITEM:
            page_parts.append("\n\n")

        page_markdown = "".join(page_parts)
        full_parts.append(page_markdown)
        raw_page: dict[str, Any] = {"page": page_index + 1, "markdown": page_markdown}
        if page_header_markdown:
            raw_page["pageHeaderMarkdown"] = page_header_markdown
        if page_footer_markdown:
            raw_page["pageFooterMarkdown"] = page_footer_markdown
        raw_pages.append(raw_page)

    return StagedDoclingMarkdown(
        markdown="".join(full_parts),
        pages=raw_pages,
        num_tables=counter_tables,
        num_pictures=counter_pictures,
    )


def stage_doc_docling(conversion_result: Any, output_path: str | Path, margin: float = 2.0) -> int:
    """Compatibility wrapper matching the standalone script's file-writing API."""

    staged = stage_docling_markdown(conversion_result, margin=margin)
    Path(output_path).write_text(staged.markdown, encoding="utf-8")
    return staged.num_tables


def _relabel_repeated_headers(doc_pages: list[Any], *, margin: float) -> None:
    unique_header_areas: set[tuple[float, float, float, float]] = set()

    for page in doc_pages:
        if not getattr(page, "assembled", None) or not getattr(page.assembled, "elements", None):
            continue

        for element in page.assembled.elements:
            if element.label != DocItemLabel.PAGE_HEADER:
                continue
            bbox = _element_bbox(element)
            if bbox is None:
                continue
            unique_header_areas.add(
                (
                    bbox.t - margin,
                    bbox.b + margin,
                    bbox.l - margin,
                    bbox.r + margin,
                )
            )

    if not unique_header_areas:
        return

    for page in doc_pages:
        if not getattr(page, "assembled", None) or not getattr(page.assembled, "elements", None):
            continue

        for element in page.assembled.elements:
            if element.label == DocItemLabel.PAGE_HEADER:
                continue
            bbox = _element_bbox(element)
            if bbox is None:
                continue

            for top, bottom, left, right in unique_header_areas:
                if bbox.t >= top and bbox.b <= bottom and bbox.l >= left and bbox.r <= right:
                    element.label = DocItemLabel.PAGE_HEADER
                    if getattr(element, "cluster", None) is not None:
                        element.cluster.label = DocItemLabel.PAGE_HEADER
                    break


def _element_bbox(element: Any) -> Any | None:
    cluster = getattr(element, "cluster", None)
    return getattr(cluster, "bbox", None) if cluster is not None else None


def _collect_page_section_markdown(elements: list[Any], label: DocItemLabel) -> str:
    parts = [
        text.strip()
        for element in elements
        if getattr(element, "label", None) == label
        if isinstance(text := getattr(element, "text", None), str) and text.strip()
    ]
    return "\n\n".join(parts)


def _build_doc_metadata(docling_doc: Any) -> dict[str, Any]:
    if docling_doc is None:
        return {
            "schema_name": "",
            "version": "",
            "name": "",
            "origin": None,
        }

    return {
        "schema_name": getattr(docling_doc, "schema_name", ""),
        "version": getattr(docling_doc, "version", ""),
        "name": getattr(docling_doc, "name", ""),
        "origin": None,
    }


def _extract_picture_descriptions(docling_doc: Any) -> list[str | None]:
    picture_descriptions: list[str | None] = []
    for picture in getattr(docling_doc, "pictures", []) or []:
        description = None
        for annotation in getattr(picture, "annotations", []) or []:
            if getattr(annotation, "kind", None) == "description":
                description = getattr(annotation, "text", None)
                break
        picture_descriptions.append(description)
    return picture_descriptions


def _export_element_table(*, element: Any, doc_metadata: dict[str, Any], table_index: int) -> str:
    try:
        table_data = TableData(
            table_cells=element.table_cells,
            num_rows=element.num_rows,
            num_cols=element.num_cols,
        )
        table_item = TableItem(
            data=table_data,
            label=element.label,
            self_ref=f"#/tables/{table_index}",
            parent=None,
            annotations=[],
        )
        html = table_item.export_to_html(add_caption=True)
        return html if isinstance(html, str) else ""
    except Exception:
        try:
            markdown = table_item.export_to_markdown(doc=doc_metadata)
            return markdown if isinstance(markdown, str) else ""
        except Exception:
            return ""
