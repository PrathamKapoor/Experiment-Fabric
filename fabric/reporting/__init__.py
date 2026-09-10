"""Scientific reporting: one document model, four renderings."""

from fabric.reporting.builder import build_report
from fabric.reporting.certificate import Certificate, issue, verify_certificate
from fabric.reporting.document import Document, Section
from fabric.reporting import html, latex, markdown, plots

__all__ = [
    "build_report",
    "Document",
    "Section",
    "Certificate",
    "issue",
    "verify_certificate",
    "markdown",
    "html",
    "latex",
    "plots",
]
