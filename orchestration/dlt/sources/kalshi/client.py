"""Kalshi API client with RSA-PSS authentication.

Carried over unchanged from the v1 warehouse. It has run against the live API
for a season — the signing scheme and the retry set are both settled. Rewriting
it buys nothing.
"""
import base64
import time
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    """HTTP client for Kalshi API with RSA-PSS signed requests and retry."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, key_id: str, private_key_pem: str,
                 timeout: int = 30, max_retries: int = 3):
        self.key_id = key_id
        self.private_key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None
        )
        self.timeout = timeout
        self.max_retries = max_retries

    def _sign(self, method: str, path: str) -> dict:
        """Build auth headers with RSA-PSS signature.

        The signed message is timestamp + method + path, with the query string
        stripped — Kalshi signs the path only, so a signature built over the
        full URL is rejected.
        """
        timestamp = str(int(time.time() * 1000))
        path_clean = path.split("?")[0]
        message = f"{timestamp}{method}{path_clean}".encode()
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    def get(self, endpoint: str, params: dict | None = None, log=None) -> dict:
        """GET with exponential backoff on transient failures.

        Retries connection errors, timeouts, and 502/503/504. Everything else —
        4xx in particular — raises immediately, since a bad signature or a bad
        parameter will not fix itself on a second attempt.
        """
        url = f"{self.BASE_URL}{endpoint}"
        path = urlparse(url).path

        for attempt in range(self.max_retries):
            try:
                headers = self._sign("GET", path)
                response = requests.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    if log:
                        log.warning(f"Retry {attempt + 1}/{self.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

            except requests.exceptions.HTTPError as e:
                if response.status_code in (502, 503, 504) and attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    if log:
                        log.warning(f"Retry {attempt + 1}/{self.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

        return {}
