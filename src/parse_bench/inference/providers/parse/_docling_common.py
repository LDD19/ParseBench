"""Common functionality for docling and docling_serve providers."""

from typing import Any

from docling_core.types.doc.document import DoclingDocument

from parse_bench.layout_label_mapping import (
    UnknownRawLayoutLabelError,
    map_docling_raw_label_to_canonical,
)
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
)

_DOCLING_EXCLUDED_LAYOUT_LABELS = frozenset(
    {
        "empty_value",
        "field_heading",
        "field_hint",
        "field_item",
        "field_key",
        "field_region",
        "field_value",
        "marker",
    }
)
_DOCLING_TABLE_LABELS = frozenset({"document_index", "table"})
_DOCLING_IMAGE_LABELS = frozenset({"chart", "picture"})
_DOCLING_PAGE_HEADER_LABEL = "page_header"
_DOCLING_PAGE_FOOTER_LABEL = "page_footer"
_DOCLING_PAGE_SECTION_LABELS = frozenset({_DOCLING_PAGE_HEADER_LABEL, _DOCLING_PAGE_FOOTER_LABEL})
_INFERRED_PAGE_HEADER_MAX_Y = 0.12
_INFERRED_PAGE_FOOTER_MIN_Y = 0.72


def _normalize_docling_label(label: object) -> str | None:
    if label is None:
        return None
    value = getattr(label, "value", label)
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def _should_include_docling_label(raw_label: str) -> bool:
    if raw_label in _DOCLING_EXCLUDED_LAYOUT_LABELS:
        return False
    try:
        map_docling_raw_label_to_canonical(raw_label)
    except UnknownRawLayoutLabelError:
        return False
    return True


def _docling_item_type(raw_label: str) -> str:
    if raw_label in _DOCLING_TABLE_LABELS:
        return "table"
    if raw_label in _DOCLING_IMAGE_LABELS:
        return "image"
    return "text"


def _extract_docling_item_value(item: Any, doc: DoclingDocument, raw_label: str) -> str:
    item_type = _docling_item_type(raw_label)
    if item_type == "image":
        return ""

    if item_type == "table" and hasattr(item, "export_to_html"):
        try:
            html = item.export_to_html(doc=doc, add_caption=True)
            if isinstance(html, str):
                return html
        except Exception:
            pass

    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text

    if hasattr(item, "export_to_markdown"):
        try:
            markdown = item.export_to_markdown()
            if isinstance(markdown, str):
                return markdown
        except Exception:
            pass

    return ""


def _normalize_docling_charspan(
    charspan: object,
    *,
    text_length: int,
    include_span: bool,
) -> tuple[int | None, int | None]:
    if not include_span or not isinstance(charspan, (list, tuple)) or len(charspan) != 2:
        return (None, None)

    start_raw, end_raw = charspan
    if not isinstance(start_raw, int) or not isinstance(end_raw, int):
        return (None, None)

    start = max(0, min(start_raw, text_length))
    end_exclusive = max(start, min(end_raw, text_length))
    if end_exclusive <= start:
        return (None, None)

    # Docling charspan behaves like a Python slice [start, end).
    return (start, end_exclusive - 1)


def _build_docling_segment(
    *,
    prov: Any,
    raw_label: str,
    page_width: float,
    page_height: float,
    include_span: bool,
    text_length: int,
) -> LayoutSegmentIR | None:
    bbox = getattr(prov, "bbox", None)
    if bbox is None or page_width <= 0 or page_height <= 0:
        return None

    bbox_top_left = bbox.to_top_left_origin(page_height=page_height)
    width = bbox_top_left.r - bbox_top_left.l
    height = bbox_top_left.b - bbox_top_left.t
    if width <= 0 or height <= 0:
        return None

    start_index, end_index = _normalize_docling_charspan(
        getattr(prov, "charspan", None),
        text_length=text_length,
        include_span=include_span,
    )

    return LayoutSegmentIR(
        x=bbox_top_left.l / page_width,
        y=bbox_top_left.t / page_height,
        w=width / page_width,
        h=height / page_height,
        confidence=1.0,
        label=raw_label,
        start_index=start_index,
        end_index=end_index,
    )


def _merge_segments(segments: list[LayoutSegmentIR]) -> LayoutSegmentIR | None:
    if not segments:
        return None

    x1 = min(segment.x for segment in segments)
    y1 = min(segment.y for segment in segments)
    x2 = max(segment.x + segment.w for segment in segments)
    y2 = max(segment.y + segment.h for segment in segments)
    return LayoutSegmentIR(
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        confidence=1.0,
        label=segments[0].label,
    )


def _item_sort_key(item: LayoutItemIR) -> tuple[float, float]:
    bbox = item.bbox
    if bbox is None:
        return (1.0, 1.0)
    return (bbox.y, bbox.x)


def _join_section_markdowns(*values: str) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        parts.append(stripped)
        seen.add(stripped)
    return "\n\n".join(parts)


def _raw_page_section_markdown(page_data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = page_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _infer_page_section_markdown_from_items(
    items: list[LayoutItemIR],
    *,
    section_label: str,
) -> str:
    entries: list[tuple[tuple[float, float], str]] = []

    for item in items:
        if item.type != "text" or item.bbox is None:
            continue
        labels = {segment.label for segment in item.layout_segments if segment.label}
        if labels & _DOCLING_PAGE_SECTION_LABELS:
            continue
        value = item.value.strip()
        if not value:
            continue

        bbox = item.bbox
        if section_label == _DOCLING_PAGE_HEADER_LABEL and bbox.y <= _INFERRED_PAGE_HEADER_MAX_Y:
            entries.append(((bbox.y, bbox.x), value))
        elif section_label == _DOCLING_PAGE_FOOTER_LABEL and bbox.y + bbox.h >= _INFERRED_PAGE_FOOTER_MIN_Y:
            entries.append(((bbox.y, bbox.x), value))

    return "\n\n".join(value for _key, value in sorted(entries))


def _build_docling_page_section_items(
    *,
    doc: DoclingDocument,
    page_number: int,
    page_width: float,
    page_height: float,
    skip_refs: set[str],
) -> tuple[str, str, list[LayoutItemIR], list[LayoutItemIR]]:
    """Build page-header/footer layout items from Docling furniture text.

    Docling's ``iterate_items(page_no=...)`` walks the document body and can omit
    furniture-layer items, so page header/footer text needs a separate pass over
    ``doc.texts``.
    """

    header_entries: list[tuple[tuple[float, float], str]] = []
    footer_entries: list[tuple[tuple[float, float], str]] = []
    header_items: list[LayoutItemIR] = []
    footer_items: list[LayoutItemIR] = []

    for item in getattr(doc, "texts", []) or []:
        raw_label = _normalize_docling_label(getattr(item, "label", None))
        if raw_label not in _DOCLING_PAGE_SECTION_LABELS:
            continue

        page_provs = [
            prov for prov in getattr(item, "prov", []) or [] if getattr(prov, "page_no", None) == page_number
        ]
        if not page_provs:
            continue

        item_value = _extract_docling_item_value(item, doc, raw_label)
        if not item_value.strip():
            continue

        segments = [
            segment
            for prov in page_provs
            if (
                segment := _build_docling_segment(
                    prov=prov,
                    raw_label=raw_label,
                    page_width=page_width,
                    page_height=page_height,
                    include_span=True,
                    text_length=len(item_value),
                )
            )
            is not None
        ]
        if not segments:
            continue

        merged_bbox = _merge_segments(segments)
        layout_item = LayoutItemIR(
            type="text",
            value=item_value,
            bbox=merged_bbox,
            layout_segments=segments,
        )
        sort_key = _item_sort_key(layout_item)
        if raw_label == _DOCLING_PAGE_HEADER_LABEL:
            header_entries.append((sort_key, item_value))
        else:
            footer_entries.append((sort_key, item_value))

        self_ref = getattr(item, "self_ref", None)
        if self_ref is not None and str(self_ref) in skip_refs:
            continue

        if raw_label == _DOCLING_PAGE_HEADER_LABEL:
            header_items.append(layout_item)
        else:
            footer_items.append(layout_item)

    header_items.sort(key=_item_sort_key)
    footer_items.sort(key=_item_sort_key)
    header_markdown = "\n\n".join(value for _key, value in sorted(header_entries))
    footer_markdown = "\n\n".join(value for _key, value in sorted(footer_entries))
    return header_markdown, footer_markdown, header_items, footer_items


def _build_docling_layout_pages(
    *,
    doc: DoclingDocument,
    raw_pages: list[dict[str, Any]],
) -> list[ParseLayoutPageIR]:
    page_markdown_by_number: dict[int, str] = {}
    page_header_by_number: dict[int, str] = {}
    page_footer_by_number: dict[int, str] = {}
    for page_data in raw_pages:
        page_number = page_data.get("page")
        if isinstance(page_number, int) and page_number > 0:
            page_markdown_by_number[page_number] = str(page_data.get("markdown", ""))
            page_header_by_number[page_number] = _raw_page_section_markdown(
                page_data,
                "pageHeaderMarkdown",
                "page_header_markdown",
            )
            page_footer_by_number[page_number] = _raw_page_section_markdown(
                page_data,
                "pageFooterMarkdown",
                "page_footer_markdown",
            )

    layout_pages: list[ParseLayoutPageIR] = []
    for page_number in sorted(doc.pages.keys()):
        page = doc.pages[page_number]
        page_width = float(page.size.width)
        page_height = float(page.size.height)
        items: list[LayoutItemIR] = []
        seen_refs: set[str] = set()

        for item, _level in doc.iterate_items(page_no=page_number):
            raw_label = _normalize_docling_label(getattr(item, "label", None))
            if raw_label is None or not _should_include_docling_label(raw_label):
                continue

            item_type = _docling_item_type(raw_label)
            item_value = _extract_docling_item_value(item, doc, raw_label)
            include_span = item_type == "text"

            page_provs = [
                prov for prov in getattr(item, "prov", []) or [] if getattr(prov, "page_no", None) == page_number
            ]
            segments = [
                segment
                for prov in page_provs
                if (
                    segment := _build_docling_segment(
                        prov=prov,
                        raw_label=raw_label,
                        page_width=page_width,
                        page_height=page_height,
                        include_span=include_span,
                        text_length=len(item_value),
                    )
                )
                is not None
            ]
            if not segments:
                continue

            merged_bbox = _merge_segments(segments)
            self_ref = getattr(item, "self_ref", None)
            if self_ref is not None:
                seen_refs.add(str(self_ref))
            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=item_value,
                    bbox=merged_bbox,
                    layout_segments=segments,
                )
            )

        doc_header_markdown, doc_footer_markdown, header_items, footer_items = _build_docling_page_section_items(
            doc=doc,
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            skip_refs=seen_refs,
        )
        items = header_items + items + footer_items

        inferred_header_markdown = _infer_page_section_markdown_from_items(
            items,
            section_label=_DOCLING_PAGE_HEADER_LABEL,
        )
        inferred_footer_markdown = _infer_page_section_markdown_from_items(
            items,
            section_label=_DOCLING_PAGE_FOOTER_LABEL,
        )
        page_header_markdown = _join_section_markdowns(
            page_header_by_number.get(page_number, ""),
            doc_header_markdown,
            inferred_header_markdown,
        )
        page_footer_markdown = _join_section_markdowns(
            page_footer_by_number.get(page_number, ""),
            doc_footer_markdown,
            inferred_footer_markdown,
        )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_number,
                width=page_width,
                height=page_height,
                md=page_markdown_by_number.get(page_number, ""),
                page_header_markdown=page_header_markdown,
                page_footer_markdown=page_footer_markdown,
                items=items,
            )
        )

    return layout_pages
