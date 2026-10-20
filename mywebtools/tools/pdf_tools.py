"""PDF tools: extract text/images (with OCR fallback), merge, split, convert, compress."""

from __future__ import annotations

import asyncio
import re
import zipfile
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .. import ocr
from ..errors import ToolError
from ..i18n import only_latin_english
from ..textclean import clean_text
from .base import Group, Param, Tool, file_param, register

PDF_ACCEPT = ".pdf,application/pdf"
IMAGE_ACCEPT = "image/*,.jpg,.jpeg,.png,.webp,.heic,.heif,.tif,.tiff,.bmp"


def _open(path: Path) -> fitz.Document:
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ToolError("pdf.unreadable", str(exc)) from exc
    if doc.needs_pass:
        doc.close()
        raise ToolError("pdf.encrypted")
    return doc


def parse_ranges(spec: str, count: int) -> list[list[int]]:
    """'1-3,5,8-' -> [[0,1,2],[4],[7..count-1]] (0-based, each part one output)."""
    parts = []
    for chunk in re.split(r"[,;\s]+", spec.strip()):
        if not chunk:
            continue
        m = re.fullmatch(r"(\d*)\s*-\s*(\d*)|(\d+)", chunk)
        if not m:
            raise ToolError("pdf.bad_range", range=chunk)
        if m.group(3):
            start = end = int(m.group(3))
        else:
            start = int(m.group(1) or 1)
            end = int(m.group(2) or count)
        start, end = max(1, start), min(count, end)
        if start > end:
            raise ToolError("pdf.bad_range", range=chunk)
        parts.append(list(range(start - 1, end)))
    if not parts:
        raise ToolError("pdf.bad_range", range=spec)
    return parts


def ocr_params(default_dates: bool = False) -> list[Param]:
    return [
        Param("ocr_langs", "multiselect", "OCR languages", default=["eng"], options=ocr.languages() or ["eng"],
              help="Installed Tesseract language packs. Combine e.g. English + Simplified Chinese.", group="ocr"),
        Param("ascii_only", "select", "Keep only ASCII characters", default="auto",
              options=[["auto", "Auto (only when OCR language is English)"], ["on", "On"], ["off", "Off"]],
              group="ocr", advanced=True),
        Param("remove_dates", "bool", "Remove dates and times", default=default_dates, group="ocr", advanced=True,
              help="Strips dates, clock times and years in English and CJK formats."),
    ]


def ascii_flag(params: dict[str, Any]) -> bool:
    mode = params.get("ascii_only", "auto")
    return mode == "on" or (mode == "auto" and only_latin_english(params.get("ocr_langs") or ["eng"]))


@register
class PdfExtract(Tool):
    id = "pdf_extract"
    category = "pdf"
    icon = "file-text"
    name = "PDF text & images"
    description = "Extract text (OCR for scanned pages) and embedded images into a ZIP."
    groups = [Group("main", ""), Group("ocr", "OCR")]

    def params(self) -> list[Param]:
        return [
            file_param("pdf", "PDF files", PDF_ACCEPT, multiple=True, batch=True),
            Param("ocr_mode", "select", "OCR", default="auto",
                  options=[["auto", "Only pages without a text layer"], ["always", "Every page"], ["never", "Never"]]),
            Param("dpi", "int", "OCR resolution", default=200, min=100, max=400, step=50, unit="dpi", group="ocr"),
            Param("images", "bool", "Extract embedded images", default=True),
            Param("per_page", "bool", "Page headers in text", default=True),
            *ocr_params(),
        ]

    def availability(self) -> tuple[bool, str]:
        return True, ""

    async def run(self, ctx, params):
        source = ctx.input("pdf")
        mode = params["ocr_mode"]
        if mode != "never" and not ocr.available():
            mode = "never"
            ctx.log("tesseract not installed: OCR disabled")
        doc = _open(source)
        try:
            count = doc.page_count
            zip_path = ctx.out_path(f"{source.stem}_extract.zip")
            texts: list[str] = []
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for index in range(count):
                    ctx.check_cancel()
                    page = doc[index]
                    text = await asyncio.to_thread(page.get_text, "text")
                    if mode == "always" or (mode == "auto" and len(text.strip()) < 40):
                        image = ctx.workdir / f"page_{index + 1}.png"
                        pix = await asyncio.to_thread(page.get_pixmap, dpi=params["dpi"], colorspace=fitz.csGRAY)
                        await asyncio.to_thread(pix.save, str(image))
                        scanned = await ocr.image_text(image, params["ocr_langs"])
                        image.unlink(missing_ok=True)
                        if len(scanned.strip()) > len(text.strip()):
                            text = scanned
                    text = clean_text(text, ascii_only=ascii_flag(params), remove_dates=params["remove_dates"])
                    if text:
                        texts.append((f"--- Page {index + 1} ---\n" if params["per_page"] else "") + text)
                    if params["images"]:
                        for n, info in enumerate(page.get_images(full=True), 1):
                            data = await asyncio.to_thread(doc.extract_image, info[0])
                            if data and data.get("image"):
                                ext = re.sub(r"[^a-z0-9]", "", data.get("ext", "bin").lower()) or "bin"
                                archive.writestr(f"images/page_{index + 1:03d}_{n}.{ext}", data["image"])
                    ctx.progress(100 * (index + 1) / count, "pdf.page", page=index + 1, pages=count)
                body = "\n\n".join(texts) + "\n"
                archive.writestr("text.txt", body)
            txt_path = ctx.out_path(f"{source.stem}.txt")
            txt_path.write_text(body, encoding="utf-8")
            ctx.add_output(zip_path, "zip")
            ctx.add_output(txt_path, "text")
            ctx.set_result({"pages": count, "characters": len(body)})
        finally:
            doc.close()


@register
class PdfMerge(Tool):
    id = "pdf_merge"
    category = "pdf"
    icon = "layers"
    name = "Merge PDFs"
    description = "Combine several PDFs into one, in the order selected."

    def params(self):
        return [
            file_param("pdfs", "PDF files", PDF_ACCEPT, multiple=True),
            Param("order", "select", "Order", default="selected",
                  options=[["selected", "As selected"], ["name", "By file name"]]),
            Param("output_name", "text", "Output file name", default="merged.pdf"),
            Param("bookmarks", "bool", "Add a bookmark per file", default=True),
        ]

    async def run(self, ctx, params):
        paths = ctx.inputs("pdfs")
        if len(paths) < 2:
            raise ToolError("pdf.need_two")
        if params["order"] == "name":
            paths.sort(key=lambda p: p.name.casefold())

        def merge() -> tuple[Path, int]:
            out = fitz.open()
            toc = []
            for path in paths:
                with _open(path) as src:
                    toc.append([1, path.stem, out.page_count + 1])
                    out.insert_pdf(src)
            if params["bookmarks"]:
                out.set_toc(toc)
            name = params["output_name"] or "merged.pdf"
            target = ctx.out_path(name if name.lower().endswith(".pdf") else name + ".pdf")
            out.save(str(target), garbage=3, deflate=True)
            pages = out.page_count
            out.close()
            return target, pages

        target, pages = await asyncio.to_thread(merge)
        ctx.add_output(target)
        ctx.set_result({"files": len(paths), "pages": pages})


@register
class PdfSplit(Tool):
    id = "pdf_split"
    category = "pdf"
    icon = "scissors"
    name = "Split PDF"
    description = "Split by page ranges or every N pages."

    def params(self):
        return [
            file_param("pdf", "PDF file", PDF_ACCEPT),
            Param("mode", "select", "Split by", default="ranges",
                  options=[["ranges", "Page ranges"], ["every", "Every N pages"]]),
            Param("ranges", "text", "Ranges", default="1-3,4-", help="e.g. 1-3,5,8- (each range = one file)",
                  show_if={"mode": "ranges"}),
            Param("every", "int", "Pages per file", default=1, min=1, max=1000, show_if={"mode": "every"}),
            Param("zip", "bool", "Bundle as ZIP", default=True),
        ]

    async def run(self, ctx, params):
        source = ctx.input("pdf")

        def split() -> list[Path]:
            with _open(source) as doc:
                count = doc.page_count
                if params["mode"] == "every":
                    n = max(1, params["every"])
                    groups = [list(range(i, min(i + n, count))) for i in range(0, count, n)]
                else:
                    groups = parse_ranges(params["ranges"], count)
                outputs = []
                for pages in groups:
                    part = fitz.open()
                    part.insert_pdf(doc, from_page=pages[0], to_page=pages[-1])
                    label = f"{pages[0] + 1}" if len(pages) == 1 else f"{pages[0] + 1}-{pages[-1] + 1}"
                    target = ctx.workdir / f"{source.stem}_p{label}.pdf"
                    part.save(str(target), garbage=3, deflate=True)
                    part.close()
                    outputs.append(target)
                return outputs

        parts = await asyncio.to_thread(split)
        if params["zip"] and len(parts) > 1:
            target = ctx.out_path(f"{source.stem}_split.zip")
            with zipfile.ZipFile(target, "w") as archive:
                for part in parts:
                    archive.write(part, part.name)
            ctx.add_output(target)
        else:
            for part in parts:
                ctx.add_output(part)
        ctx.set_result({"parts": len(parts)})


@register
class PdfToImages(Tool):
    id = "pdf_to_images"
    category = "pdf"
    icon = "image"
    name = "PDF to images"
    description = "Render every page as PNG or JPEG."

    def params(self):
        return [
            file_param("pdf", "PDF files", PDF_ACCEPT, multiple=True, batch=True),
            Param("format", "select", "Format", default="png", options=[["png", "PNG"], ["jpg", "JPEG"]]),
            Param("dpi", "int", "Resolution", default=150, min=50, max=600, step=25, unit="dpi"),
            Param("pages", "text", "Pages", default="", help="Empty = all pages, or e.g. 1-3,5"),
        ]

    async def run(self, ctx, params):
        source = ctx.input("pdf")
        doc = _open(source)
        try:
            pages = [p for group in parse_ranges(params["pages"], doc.page_count) for p in group] \
                if params["pages"].strip() else list(range(doc.page_count))
            target = ctx.out_path(f"{source.stem}_pages.zip")
            with zipfile.ZipFile(target, "w") as archive:
                for n, index in enumerate(pages, 1):
                    ctx.check_cancel()
                    pix = await asyncio.to_thread(doc[index].get_pixmap, dpi=params["dpi"])
                    data = await asyncio.to_thread(pix.tobytes, "jpg" if params["format"] == "jpg" else "png")
                    archive.writestr(f"{source.stem}_{index + 1:03d}.{params['format']}", data)
                    ctx.progress(100 * n / len(pages), "pdf.page", page=index + 1, pages=len(pages))
            ctx.add_output(target)
        finally:
            doc.close()


@register
class ImagesToPdf(Tool):
    id = "images_to_pdf"
    category = "pdf"
    icon = "file-plus"
    name = "Images to PDF"
    description = "Turn photos or scans (incl. iPhone HEIC) into one PDF."

    def params(self):
        return [
            file_param("images", "Images", IMAGE_ACCEPT, multiple=True),
            Param("page_size", "select", "Page size", default="fit",
                  options=[["fit", "Same as image"], ["a4", "A4"], ["letter", "Letter"]]),
            Param("margin", "int", "Margin", default=0, min=0, max=200, unit="pt", show_if={"page_size": ["a4", "letter"]}),
            Param("quality", "int", "JPEG quality", default=85, min=40, max=100),
            Param("max_side", "int", "Max image side", default=2400, min=600, max=8000, unit="px"),
            Param("output_name", "text", "Output file name", default="images.pdf"),
        ]

    async def run(self, ctx, params):
        from PIL import Image, ImageOps

        try:
            import pillow_heif  # type: ignore

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        paths = ctx.inputs("images")

        def build() -> Path:
            doc = fitz.open()
            sizes = {"a4": fitz.paper_rect("a4"), "letter": fitz.paper_rect("letter")}
            for path in paths:
                try:
                    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                except Exception as exc:  # noqa: BLE001
                    raise ToolError("image.unreadable", str(exc), name=path.name) from exc
                img.thumbnail((params["max_side"], params["max_side"]))
                jpg = ctx.workdir / (path.stem + ".jpg")
                img.save(jpg, "JPEG", quality=params["quality"])
                rect = sizes.get(params["page_size"]) or fitz.Rect(0, 0, img.width * 0.75, img.height * 0.75)
                page = doc.new_page(width=rect.width, height=rect.height)
                m = params["margin"] if params["page_size"] != "fit" else 0
                page.insert_image(fitz.Rect(m, m, rect.width - m, rect.height - m), filename=str(jpg), keep_proportion=True)
            name = params["output_name"] or "images.pdf"
            target = ctx.out_path(name if name.lower().endswith(".pdf") else name + ".pdf")
            doc.save(str(target), garbage=3, deflate=True)
            doc.close()
            return target

        ctx.add_output(await asyncio.to_thread(build))


@register
class PdfCompress(Tool):
    id = "pdf_compress"
    category = "pdf"
    icon = "minimize"
    name = "Compress PDF"
    description = "Recompress images and clean up the file structure."

    def params(self):
        return [
            file_param("pdf", "PDF files", PDF_ACCEPT, multiple=True, batch=True),
            Param("image_dpi", "int", "Downsample images to", default=150, min=0, max=600, step=25, unit="dpi",
                  help="0 = keep image resolution"),
            Param("quality", "int", "JPEG quality", default=75, min=30, max=95),
        ]

    async def run(self, ctx, params):
        source = ctx.input("pdf")

        def compress() -> Path:
            with _open(source) as doc:
                if params["image_dpi"] and hasattr(doc, "rewrite_images"):
                    doc.rewrite_images(dpi_threshold=params["image_dpi"] + 10, dpi_target=params["image_dpi"],
                                       quality=params["quality"])
                target = ctx.out_path(f"{source.stem}_compressed.pdf")
                doc.save(str(target), garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, clean=True)
                return target

        target = await asyncio.to_thread(compress)
        before, after = source.stat().st_size, target.stat().st_size
        ctx.add_output(target)
        ctx.set_result({"before": before, "after": after, "saved_percent": round(100 * (1 - after / before), 1) if before else 0})
