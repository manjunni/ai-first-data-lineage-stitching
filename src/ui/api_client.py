from __future__ import annotations

import requests
from typing import Optional


class LineageApiClient:
    """Thin synchronous wrapper around the FastAPI backend."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return self._get("/health")

    def list_runs(self, status: Optional[str] = None) -> list[dict]:
        params = {"status": status} if status else {}
        return self._get("/runs", params=params) or []

    def start_run(self, files: list[tuple], catalog: str = "dry_run",
                  artifact_types: str = "") -> dict:
        data = {"catalog": catalog, "artifact_types": artifact_types}
        return self._post_multipart("/runs", files=files, data=data)

    def get_run(self, run_id: str) -> Optional[dict]:
        return self._get(f"/runs/{run_id}")

    # ------------------------------------------------------------------
    # Lineage graph
    # ------------------------------------------------------------------

    def get_lineage(self, run_id: str) -> Optional[dict]:
        return self._get(f"/runs/{run_id}/lineage")

    # ------------------------------------------------------------------
    # Gaps and questions
    # ------------------------------------------------------------------

    def get_gaps(self, run_id: str, resolved: Optional[bool] = None) -> Optional[dict]:
        params = {} if resolved is None else {"resolved": str(resolved).lower()}
        return self._get(f"/runs/{run_id}/gaps", params=params)

    def get_questions(self, run_id: str, answered: Optional[bool] = None) -> Optional[dict]:
        params = {} if answered is None else {"answered": str(answered).lower()}
        return self._get(f"/runs/{run_id}/questions", params=params)

    # ------------------------------------------------------------------
    # SME responses
    # ------------------------------------------------------------------

    def submit_responses(self, run_id: str, responses: list[dict]) -> Optional[dict]:
        return self._post(f"/runs/{run_id}/responses", {"responses": responses})

    # ------------------------------------------------------------------
    # Visuals
    # ------------------------------------------------------------------

    def get_visuals(self, run_id: str) -> Optional[dict]:
        return self._get(f"/runs/{run_id}/visuals")

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def get_confidence(self, run_id: str) -> Optional[dict]:
        return self._get(f"/runs/{run_id}/confidence")

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_audit(self, run_id: str) -> Optional[dict]:
        return self._get(f"/runs/{run_id}/audit")

    # ------------------------------------------------------------------
    # Catalog publishing
    # ------------------------------------------------------------------

    def publish_preview(self, run_id: str, catalog: str = "alation") -> Optional[dict]:
        return self._get(f"/runs/{run_id}/publish-preview", params={"catalog": catalog})

    def publish(self, run_id: str, catalog: str = "dry_run") -> Optional[dict]:
        return self._post(f"/runs/{run_id}/publish", {"catalog": catalog})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> Optional[dict | list]:
        try:
            resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            raise ApiError(exc.response.status_code, exc.response.text) from exc
        except Exception as exc:
            raise ApiError(0, str(exc)) from exc

    def _post(self, path: str, body: dict) -> Optional[dict]:
        try:
            resp = self._session.post(f"{self.base_url}{path}", json=body, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            raise ApiError(exc.response.status_code, exc.response.text) from exc
        except Exception as exc:
            raise ApiError(0, str(exc)) from exc

    def _post_multipart(self, path: str, files: list, data: dict) -> Optional[dict]:
        try:
            resp = self._session.post(
                f"{self.base_url}{path}", files=files, data=data, timeout=120
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            raise ApiError(exc.response.status_code, exc.response.text) from exc
        except Exception as exc:
            raise ApiError(0, str(exc)) from exc


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")
