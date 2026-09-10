"""Scientific reporting: one document model, four renderings."""

from fabric.reporting import html, latex, markdown, plots
from fabric.reporting.builder import build_report
from fabric.reporting.certificate import Certificate, issue, verify_certificate
from fabric.reporting.document import Document, Section

__all__ = [
    "Certificate",
    "Document",
    "Section",
    "build_report",
    "html",
    "issue",
    "latex",
    "markdown",
    "plots",
    "verify_certificate",
]
