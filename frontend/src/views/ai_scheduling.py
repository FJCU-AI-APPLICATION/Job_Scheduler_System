"""AI scheduling tab: compute → preview → confirm."""

import json

import gradio as gr

from api_client import client
from api_client.client import BackendError


def _compute(policy_id, employee_ids_csv, start_date, end_date):
    if not policy_id or not employee_ids_csv or not start_date or not end_date:
        gr.Warning("請填寫政策 ID、員工 ID、起訖日期")
        return "", ""
    try:
        emp_ids = [int(x.strip()) for x in employee_ids_csv.split(",") if x.strip()]
    except ValueError:
        gr.Warning("員工 ID 必須是逗號分隔的整數")
        return "", ""
    try:
        result = client.compute_schedule(
            policy_id=int(policy_id),
            employee_ids=emp_ids,
            start_date=start_date,
            end_date=end_date,
        )
    except (BackendError, ValueError) as e:
        gr.Warning(f"計算失敗: {e}")
        return "", ""
    summary = (
        f"政策 #{result['policy_id']}, "
        f"{result['start_date']} ~ {result['end_date']}, "
        f"{len(result.get('schedule', []))} 天, "
        f"{len(result.get('shift_details', []))} 班別/天"
    )
    return summary, json.dumps(result, ensure_ascii=False, indent=2)


def _confirm(computed_json):
    if not computed_json:
        gr.Warning("先計算班表")
        return ""
    try:
        result = json.loads(computed_json)
    except json.JSONDecodeError as e:
        gr.Warning(f"JSON 解析失敗: {e}")
        return ""
    payload = {
        "policy_id": result["policy_id"],
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "schedule": result["schedule"],
    }
    try:
        confirm = client.confirm_schedule(payload)
    except BackendError as e:
        gr.Warning(f"確認失敗: {e}")
        return ""
    ids = confirm.get("created_schedule_ids", [])
    gr.Info(f"已建立 {len(ids)} 筆班表")
    return f"建立的 schedule IDs: {ids}"


def render() -> None:
    gr.Markdown("## AI 排班")
    gr.Markdown(
        "輸入排班政策、員工清單、起訖日期，後端會根據員工的不可用日期計算班表。"
        "預覽 JSON 後可按「確認入庫」將結果寫入資料庫。"
    )

    with gr.Row():
        policy_id = gr.Number(label="政策 ID", precision=0)
        employee_ids = gr.Textbox(
            label="員工 IDs (逗號分隔)", placeholder="1,2,3"
        )
    with gr.Row():
        start_date = gr.Textbox(label="開始日期 (YYYY-MM-DD)")
        end_date = gr.Textbox(label="結束日期 (YYYY-MM-DD)")

    compute_btn = gr.Button("AI 計算班表", variant="primary")
    summary = gr.Markdown()
    computed = gr.Code(label="計算結果 (JSON)", language="json", lines=15)

    compute_btn.click(
        _compute,
        [policy_id, employee_ids, start_date, end_date],
        [summary, computed],
    )

    confirm_btn = gr.Button("確認入庫", variant="primary")
    confirm_result = gr.Markdown()
    confirm_btn.click(_confirm, computed, confirm_result)
