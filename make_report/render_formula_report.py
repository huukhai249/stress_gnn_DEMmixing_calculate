from __future__ import annotations

import json
from html import escape
from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle


def style_map() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#334155"),
            spaceAfter=16,
        ),
        "section": ParagraphStyle(
            "section",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "subsection": ParagraphStyle(
            "subsection",
            parent=sample["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1D4ED8"),
            spaceBefore=4,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#111827"),
            spaceAfter=4,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.7,
            leading=10.5,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=10.2,
            textColor=colors.HexColor("#111827"),
        ),
        "table_code": ParagraphStyle(
            "table_code",
            parent=sample["BodyText"],
            fontName="Courier",
            fontSize=7.9,
            leading=9.6,
            textColor=colors.HexColor("#0F172A"),
        ),
    }


def make_table(rows: list[list[Paragraph]], widths: list[float]) -> LongTable:
    table = LongTable(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#1E40AF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def render_equation_image(temp_dir: Path, label: str, latex: str) -> Image:
    latex = latex.replace(r"\boldsymbol{", "{")
    image_path = temp_dir / f"{label}.png"
    math_to_image(
        f"${latex}$",
        str(image_path),
        dpi=220,
        format="png",
        prop=FontProperties(size=15),
    )
    width_px, height_px = ImageReader(str(image_path)).getSize()
    image = Image(str(image_path))
    max_width = 170 * mm
    scale = min(1.0, max_width / float(width_px))
    image.drawWidth = width_px * scale
    image.drawHeight = height_px * scale
    return image


def page_decor(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#E2E8F0"))
    canvas.rect(0, height - 14 * mm, width, 14 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#0F172A"))
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(16 * mm, height - 9.5 * mm, doc.title)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawRightString(width - 16 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_tex(report: dict[str, object], output_path: Path) -> None:
    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amssymb,booktabs,longtable,array}",
        r"\usepackage{xcolor}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\renewcommand{\arraystretch}{1.2}",
        "",
        rf"\title{{{tex_escape(str(report['title']))}}}",
        rf"\author{{Generated from {tex_escape(str(report['script_name']))}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        rf"\noindent {tex_escape(str(report['subtitle']))}\par",
        r"\medskip",
        rf"\noindent {tex_escape('This document summarizes the exact formulas currently implemented in ' + str(report['script_name']) + ' to produce exported CSV data.')}",
        "",
        r"\section*{Output files}",
        r"\begin{longtable}{p{0.18\textwidth}p{0.32\textwidth}p{0.42\textwidth}}",
        r"\toprule",
        r"\textbf{Type} & \textbf{File name pattern} & \textbf{Content}\\",
        r"\midrule",
        r"\endhead",
    ]
    for label, file_name, description in report["output_files"]:
        lines.append(rf"{tex_escape(label)} & \texttt{{{tex_escape(file_name)}}} & {tex_escape(description)}\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", "", r"\section*{Notation}",
                  r"\begin{longtable}{p{0.22\textwidth}p{0.70\textwidth}}", r"\toprule",
                  r"\textbf{Symbol} & \textbf{Meaning}\\", r"\midrule", r"\endhead"])
    for symbol, meaning in report["notation"]:
        lines.append(rf"${symbol.strip('$')}$ & {tex_escape(meaning)}\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", "", r"\section*{Core equations}"])
    for equation in report["equations"]:
        lines.extend([
            rf"\paragraph{{{tex_escape(str(equation['label']) + '. ' + str(equation['title']))}}}",
            r"\[",
            str(equation["latex"]),
            r"\]",
            tex_escape(str(equation["description"])),
            "",
        ])
    for section_title, rows_key in [("Summary CSV fields", "summary_fields"), ("Extracted CSV fields", "extract_fields")]:
        lines.extend([
            rf"\section*{{{section_title}}}",
            r"\begin{longtable}{p{0.28\textwidth}p{0.27\textwidth}p{0.35\textwidth}}",
            r"\toprule",
            r"\textbf{Field} & \textbf{Calculation / reference} & \textbf{Meaning}\\",
            r"\midrule",
            r"\endhead",
        ])
        for field_name, calc_text, meaning in report[rows_key]:
            lines.append(rf"\texttt{{{tex_escape(field_name)}}} & {tex_escape(calc_text)} & {tex_escape(meaning)}\\")
        lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    lines.extend([r"\section*{Implementation notes}"])
    for note in report["notes"]:
        lines.append(rf"\noindent - {tex_escape(note)}\par")
    lines.extend(["", r"\end{document}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_pdf(report: dict[str, object], output_path: Path) -> None:
    report_styles = style_map()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=22 * mm,
        bottomMargin=16 * mm,
        title=str(report["title"]),
    )
    story = [
        Paragraph(escape(str(report["title"])), report_styles["title"]),
        Paragraph(escape(str(report["subtitle"])), report_styles["subtitle"]),
        Paragraph(
            escape(
                "This document summarizes the exact formulas currently implemented in "
                f"{report['script_name']} to produce exported CSV data."
            ),
            report_styles["body"],
        ),
        Spacer(1, 6),
        Paragraph("Output files", report_styles["section"]),
    ]
    output_rows = [[Paragraph("Type", report_styles["table_header"]),
                    Paragraph("File name pattern", report_styles["table_header"]),
                    Paragraph("Content", report_styles["table_header"])]]
    for label, file_name, description in report["output_files"]:
        output_rows.append([Paragraph(escape(label), report_styles["table_cell"]),
                            Paragraph(escape(file_name), report_styles["table_code"]),
                            Paragraph(escape(description), report_styles["table_cell"])])
    story.append(make_table(output_rows, [28 * mm, 72 * mm, 76 * mm]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Notation", report_styles["section"]))
    notation_rows = [[Paragraph("Symbol", report_styles["table_header"]),
                      Paragraph("Meaning", report_styles["table_header"])]]
    for symbol, meaning in report["notation"]:
        notation_rows.append([Paragraph(escape(symbol), report_styles["table_code"]),
                              Paragraph(escape(meaning), report_styles["table_cell"])])
    story.append(make_table(notation_rows, [42 * mm, 134 * mm]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Core equations", report_styles["section"]))
    temp_dir = output_path.with_name(f"_{output_path.stem}_assets")
    temp_dir.mkdir(exist_ok=True)
    for equation in report["equations"]:
        story.append(Paragraph(escape(f"{equation['label']}. {equation['title']}"), report_styles["subsection"]))
        story.append(render_equation_image(temp_dir, str(equation["label"]), str(equation["latex"])))
        story.append(Spacer(1, 3))
        story.append(Paragraph(escape(str(equation["description"])), report_styles["body"]))
        story.append(Spacer(1, 6))
    for section_title, rows_key in [("Summary CSV fields", "summary_fields"), ("Extracted CSV fields", "extract_fields")]:
        story.append(Paragraph(section_title, report_styles["section"]))
        rows = [[Paragraph("Field", report_styles["table_header"]),
                 Paragraph("Calculation / reference", report_styles["table_header"]),
                 Paragraph("Meaning", report_styles["table_header"])]]
        for field_name, calc_text, meaning in report[rows_key]:
            rows.append([Paragraph(escape(field_name), report_styles["table_code"]),
                         Paragraph(escape(calc_text), report_styles["table_cell"]),
                         Paragraph(escape(meaning), report_styles["table_cell"])])
        story.append(make_table(rows, [52 * mm, 56 * mm, 68 * mm]))
        story.append(Spacer(1, 10))
    story.append(Paragraph("Implementation notes", report_styles["section"]))
    for note in report["notes"]:
        story.append(Paragraph(escape(f"- {note}"), report_styles["body"]))
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python render_formula_report.py <report_json>")

    json_path = Path(sys.argv[1]).resolve()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    tex_path = json_path.with_suffix(".tex")
    pdf_path = json_path.with_suffix(".pdf")
    build_tex(report, tex_path)
    build_pdf(report, pdf_path)
    print(f"Wrote {tex_path.name}")
    print(f"Wrote {pdf_path.name}")


if __name__ == "__main__":
    main()
