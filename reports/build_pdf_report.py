from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "milan_traffic_forecasting_report.pdf"
FIGURES = ROOT / "outputs" / "figures"


def build_pdf() -> None:
    report = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=18,
        textColor=colors.HexColor("#1f3b5b"),
    )

    story = []
    story.append(Paragraph("Milan Telecom Traffic Forecasting", title_style))
    story.append(Paragraph("This report describes a small empirical forecasting study using the Milan Telecom Traffic 2013 dataset."))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Key findings"))
    summary_table = Table([
        ["Model", "MAE", "RMSE", "MAPE"],
        ["Random Forest", "58,206", "68,691", "50.34%"],
        ["Linear Regression", "61,880", "74,288", "56.44%"],
        ["HistGradientBoosting", "111,965", "123,747", "112.08%"],
    ], colWidths=[2.2 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Top sampled squares"))
    top_table = Table([
        ["Rank", "Square ID", "Total traffic"],
        ["1", "5161", "14,614,418"],
        ["2", "5059", "13,680,137"],
        ["3", "5259", "12,259,581"],
        ["4", "5061", "11,228,350"],
        ["5", "6064", "10,658,021"],
    ], colWidths=[0.8 * inch, 1.2 * inch, 2.1 * inch])
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 0.2 * inch))

    for fig_name in ["top_areas_by_total_traffic.png", "selected_square_daily_traffic.png"]:
        image_path = FIGURES / fig_name
        if image_path.exists():
            story.append(Paragraph(f"Figure: {fig_name}"))
            story.append(Image(str(image_path), width=5.5 * inch, height=3.2 * inch))
            story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Conclusion"))
    story.append(Paragraph("The project finds that a few highly active network cells dominate the overall traffic profile, and that a nonlinear tree-based model provides the strongest baseline for forecasting the busiest square. Future work should expand feature engineering, compare against time-series-specific methods, and evaluate multiple cells in parallel."))

    report.build(story)
    print(f"PDF saved to {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
