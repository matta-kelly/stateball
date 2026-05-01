"""Secret loading.

In production these env vars are injected by the deployment platform
(ExternalSecret + HelmRelease in the original setup; equivalent in
others). Locally, populate them from a `.env` file or shell — see
`.env.example` at the repo root for the full list.
"""

from __future__ import annotations


def load_secrets() -> None:
    """No-op stub.

    Kept so callers can import + invoke without conditional logic;
    the real wiring is environment-driven.
    """
    return
