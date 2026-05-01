"""MLB Stats API client with retry."""
import time

import requests


class MLBClient:
    """HTTP client for MLB Stats API with retry on transient errors."""

    BASE_URL = "https://statsapi.mlb.com"

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

    def get(self, endpoint: str, params: dict = None, log=None) -> dict:
        """GET request with retry on transient errors."""
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    if log:
                        log.warning(f"Retry {attempt + 1}/{self.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

            except requests.exceptions.HTTPError as e:
                if response.status_code in (502, 503, 504) and attempt < self.max_retries - 1:
                    wait = 2**attempt
                    if log:
                        log.warning(f"Retry {attempt + 1}/{self.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

        return {}
