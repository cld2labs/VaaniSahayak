"""Keycloak client credentials token manager.

Fetches and auto-refreshes Bearer tokens for the Enterprise Inference (EI)
APISIX gateway using the OAuth2 client_credentials flow.

Usage:
    from backend.services.keycloak_auth import KeycloakTokenManager

    mgr = KeycloakTokenManager(
        keycloak_url="https://keycloak.example.com",
        realm="ei-realm",
        client_id="vaani-client",
        client_secret="...",
    )
    token = mgr.get_token()   # cached; auto-refreshes when near expiry
    headers = mgr.auth_headers()  # {"Authorization": "Bearer <token>"}
"""
import logging
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Refresh the token this many seconds before it actually expires
_REFRESH_BUFFER_S = 30


class KeycloakTokenManager:
    """Thread-safe Keycloak client credentials token manager."""

    def __init__(
        self,
        keycloak_url: str,
        realm: str,
        client_id: str,
        client_secret: str,
        verify_ssl: bool = True,
    ) -> None:
        self._token_url = (
            f"{keycloak_url.rstrip('/')}/realms/{realm}"
            "/protocol/openid-connect/token"
        )
        self._client_id = client_id
        self._client_secret = client_secret
        self._verify_ssl = verify_ssl

        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0  # epoch seconds

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_token(self) -> str:
        """Return a valid Bearer token, fetching/refreshing as needed."""
        with self._lock:
            if self._needs_refresh():
                self._fetch_token()
            return self._access_token  # type: ignore[return-value]

    def auth_headers(self) -> dict[str, str]:
        """Return Authorization header dict ready to pass to httpx."""
        return {"Authorization": f"Bearer {self.get_token()}"}

    def invalidate(self) -> None:
        """Force the next call to fetch a fresh token (e.g. after 401)."""
        with self._lock:
            self._expires_at = 0.0

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _needs_refresh(self) -> bool:
        return time.time() >= (self._expires_at - _REFRESH_BUFFER_S)

    def _fetch_token(self) -> None:
        """POST to Keycloak token endpoint and cache the result."""
        try:
            r = httpx.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                verify=self._verify_ssl,
                timeout=15,
            )
            r.raise_for_status()
            payload = r.json()
            self._access_token = payload["access_token"]
            expires_in = int(payload.get("expires_in", 300))
            self._expires_at = time.time() + expires_in
            logger.info(
                "[Keycloak] Token acquired (expires in %ds)", expires_in
            )
        except Exception as exc:
            logger.error("[Keycloak] Token fetch failed: %s", exc)
            raise
