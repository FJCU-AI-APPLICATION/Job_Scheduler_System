"""Schedule view tab: list saved schedules."""

import gradio as gr

from frontend.api_client import client
from frontend.api_client.client import BackendError


def _fetch_schedules(page: int = 1, page_size: int = 20) -> tuple[list[list], str]:
    try:
        result = client.list_schedules(page=int(page or 1), page_size=int(page_size or 20))
    except (BackendError, ValueError) as e:
        gr.Warning(str(e))
        return [], "0 筆"
    rows = [
        [
            r["id"],
            r["name"],
            r["description"],
            r["start_date"],
            r["end_date"],
            r.get("start_time") or "",
            r.get("end_time") or "",
        ]
        for r in result.get("results", [])
    ]
    return rows, f"共 {result.get('count', 0)} 筆"


def render() -> None:
    gr.Markdown("## 班表查詢")

    with gr.Row():
        page = gr.Number(label="頁碼", value=1, precision=0)
        page_size = gr.Number(label="每頁筆數", value=20, precision=0)
        refresh_btn = gr.Button("查詢", variant="primary")

    count_label = gr.Markdown("")
    schedules_table = gr.Dataframe(
        headers=["id", "name", "description", "start_date", "end_date", "start_time", "end_time"],
        interactive=False,
        wrap=True,
    )

    refresh_btn.click(_fetch_schedules, [page, page_size], [schedules_table, count_label])

    initial_rows, initial_count = _fetch_schedules()
    schedules_table.value = initial_rows
    count_label.value = initial_count
