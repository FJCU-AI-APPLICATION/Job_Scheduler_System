"""Thin httpx wrapper around the backend's REST API."""

from typing import Any

import httpx

from frontend.core.config import settings


class BackendError(RuntimeError):
    """Raised when the backend returns a non-2xx response."""


class BackendClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self._client = httpx.Client(
            base_url=base_url or settings.BACKEND_URL,
            timeout=timeout or settings.REQUEST_TIMEOUT,
        )

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            r = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise BackendError(f"{method} {path} failed: {e}") from e
        if r.status_code >= 400:
            raise BackendError(f"{method} {path} -> {r.status_code}: {r.text}")
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    def get(self, path: str, **params) -> Any:
        return self._request("GET", path, params=params or None)

    def post(self, path: str, json: dict | None = None, **params) -> Any:
        return self._request("POST", path, json=json, params=params or None)

    def put(self, path: str, json: dict | None = None) -> Any:
        return self._request("PUT", path, json=json)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # Domain helpers

    def list_employees(self, page: int = 1, page_size: int = 20) -> dict:
        return self.get("/api/employee/", page=page, page_size=page_size)

    def create_employee(
        self,
        name: str,
        age: int,
        phone: str,
        identity: str,
        salary_type: str,
    ) -> dict:
        return self.post(
            "/api/employee/",
            json={
                "name": name,
                "age": age,
                "phone": phone,
                "identity": identity,
                "salary_type": salary_type,
            },
        )

    def delete_employee(self, employee_id: int) -> None:
        self.delete(f"/api/employee/{employee_id}/")

    def list_unavailabilities(self, employee_id: int) -> list[dict]:
        return self.get("/api/employee/unavailabilities/", employee_id=employee_id)

    def create_unavailability(self, payload: dict) -> dict:
        return self.post("/api/employee/unavailabilities/", json=payload)

    def list_policies(self) -> list[dict]:
        return self.get("/api/policy/")

    def create_policy(self, policy_name: str, description: str | None) -> dict:
        return self.post(
            "/api/policy/",
            json={"policy_name": policy_name, "description": description},
        )

    def list_shift_policies(self, policy_id: int) -> list[dict]:
        return self.get("/api/policy/shiftpolicy/", policy_id=policy_id)

    def create_shift_policy(self, policy_id: int, start_time: str, end_time: str) -> dict:
        return self.post(
            "/api/policy/shiftpolicy/",
            json={
                "policy_id": policy_id,
                "start_time": start_time,
                "end_time": end_time,
            },
        )

    def list_schedules(self, page: int = 1, page_size: int = 20) -> dict:
        return self.get("/api/schedule/", page=page, page_size=page_size)

    def compute_schedule(
        self,
        policy_id: int,
        employee_ids: list[int],
        start_date: str,
        end_date: str,
    ) -> dict:
        return self.post(
            "/api/schedule/compute/",
            json={
                "policy_id": policy_id,
                "employee_ids": employee_ids,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    def confirm_schedule(self, payload: dict) -> dict:
        return self.post("/api/schedule/confirm/", json=payload)


client = BackendClient()
