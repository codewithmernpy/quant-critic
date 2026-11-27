"""HTML, JSON, and Plotly reporting utilities."""

from quantcritic.report.html import (
    build_report_payload,
    generate_audit_reports,
    generate_leaderboard_report,
    render_audit_html_from_payload,
)

__all__ = [
    "build_report_payload",
    "generate_audit_reports",
    "generate_leaderboard_report",
    "render_audit_html_from_payload",
]
