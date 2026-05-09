"""Policy management tab: list policies + add shifts."""

import gradio as gr

from api_client import client
from api_client.client import BackendError


def _fetch_policies() -> list[list]:
    try:
        rows = client.list_policies()
    except BackendError as e:
        gr.Warning(str(e))
        return []
    return [[r["id"], r["policy_name"], r.get("description") or ""] for r in rows]


def _create_policy(name, description):
    if not name:
        gr.Warning("請輸入政策名稱")
        return _fetch_policies()
    try:
        client.create_policy(name, description or None)
        gr.Info(f"已新增政策 {name}")
    except BackendError as e:
        gr.Warning(f"新增失敗: {e}")
    return _fetch_policies()


def _list_shifts(policy_id):
    if not policy_id:
        return []
    try:
        rows = client.list_shift_policies(int(policy_id))
    except (BackendError, ValueError) as e:
        gr.Warning(str(e))
        return []
    return [[r["id"], r["start_time"], r["end_time"]] for r in rows]


def _add_shift(policy_id, start_time, end_time):
    if not policy_id or not start_time or not end_time:
        gr.Warning("請輸入政策 ID、開始時間、結束時間")
        return _list_shifts(policy_id)
    try:
        client.create_shift_policy(int(policy_id), start_time, end_time)
        gr.Info("已新增班別")
    except (BackendError, ValueError) as e:
        gr.Warning(f"新增失敗: {e}")
    return _list_shifts(policy_id)


def render() -> None:
    gr.Markdown("## 排班政策")

    policies_table = gr.Dataframe(
        headers=["id", "policy_name", "description"],
        value=_fetch_policies(),
        interactive=False,
        wrap=True,
    )
    refresh_btn = gr.Button("重新整理")
    refresh_btn.click(_fetch_policies, outputs=policies_table)

    with gr.Accordion("新增政策", open=False):
        name = gr.Textbox(label="政策名稱")
        desc = gr.Textbox(label="說明 (選填)", lines=2)
        create_btn = gr.Button("新增", variant="primary")
        create_btn.click(_create_policy, [name, desc], policies_table)

    with gr.Accordion("管理班別 (政策下的 shift)", open=False):
        policy_id = gr.Number(label="政策 ID", precision=0)
        shifts_table = gr.Dataframe(
            headers=["id", "start_time", "end_time"],
            interactive=False,
        )
        list_btn = gr.Button("查詢班別")
        list_btn.click(_list_shifts, policy_id, shifts_table)

        with gr.Row():
            start_time = gr.Textbox(label="開始時間 (HH:MM:SS)")
            end_time = gr.Textbox(label="結束時間 (HH:MM:SS)")
        add_btn = gr.Button("新增班別", variant="primary")
        add_btn.click(_add_shift, [policy_id, start_time, end_time], shifts_table)
