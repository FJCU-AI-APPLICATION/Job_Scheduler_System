"""Employee management tab: list, create, delete employees + unavailabilities."""

import gradio as gr

from api_client import client
from api_client.client import BackendError


def _fetch_table() -> list[list]:
    try:
        result = client.list_employees(page=1, page_size=100)
    except BackendError as e:
        gr.Warning(f"無法載入員工資料: {e}")
        return []
    return [
        [r["id"], r["name"], r["age"], r["phone"], r["identity"], r["salary_type"]]
        for r in result.get("results", [])
    ]


def _create(name, age, phone, identity, salary_type):
    if not name or not phone:
        gr.Warning("姓名和電話為必填")
        return _fetch_table()
    try:
        client.create_employee(name, int(age), phone, identity, salary_type)
        gr.Info(f"已新增員工 {name}")
    except (BackendError, ValueError) as e:
        gr.Warning(f"新增失敗: {e}")
    return _fetch_table()


def _delete(employee_id):
    if not employee_id:
        gr.Warning("請輸入員工編號")
        return _fetch_table()
    try:
        client.delete_employee(int(employee_id))
        gr.Info(f"已刪除員工 {employee_id}")
    except (BackendError, ValueError) as e:
        gr.Warning(f"刪除失敗: {e}")
    return _fetch_table()


def _list_unavail(employee_id):
    if not employee_id:
        return []
    try:
        rows = client.list_unavailabilities(int(employee_id))
    except (BackendError, ValueError) as e:
        gr.Warning(str(e))
        return []
    return [
        [
            r["id"],
            r["unavailability_type"],
            r["day_of_week"],
            r["start_date"],
            r["end_date"],
            r["reason"],
        ]
        for r in rows
    ]


def _add_unavail(employee_id, unavail_type, day_of_week, start_date, end_date, reason):
    if not employee_id:
        gr.Warning("請輸入員工編號")
        return _list_unavail(employee_id)
    payload = {
        "employee_id": int(employee_id),
        "unavailability_type": unavail_type,
        "reason": reason or "",
    }
    if unavail_type == "DAY_OF_WEEK":
        if day_of_week is None or day_of_week == "":
            gr.Warning("DAY_OF_WEEK 需要 day_of_week (1=Mon..7=Sun)")
            return _list_unavail(employee_id)
        payload["day_of_week"] = int(day_of_week)
    else:
        if not start_date or not end_date:
            gr.Warning("DATE_RANGE 需要 start_date 與 end_date")
            return _list_unavail(employee_id)
        payload["start_date"] = start_date
        payload["end_date"] = end_date
    try:
        client.create_unavailability(payload)
        gr.Info("已新增不可用紀錄")
    except BackendError as e:
        gr.Warning(f"新增失敗: {e}")
    return _list_unavail(employee_id)


def render() -> None:
    gr.Markdown("## 員工管理")

    employees_table = gr.Dataframe(
        headers=["id", "姓名", "年齡", "電話", "身份", "薪資類型"],
        value=_fetch_table(),
        interactive=False,
        wrap=True,
    )
    refresh_btn = gr.Button("重新整理")
    refresh_btn.click(_fetch_table, outputs=employees_table)

    with gr.Accordion("新增員工", open=False):
        with gr.Row():
            name = gr.Textbox(label="姓名")
            age = gr.Number(label="年齡", value=25, precision=0)
            phone = gr.Textbox(label="電話")
            identity = gr.Dropdown(["FULL", "PART"], value="FULL", label="身份")
            salary_type = gr.Dropdown(["MONTH", "HOUR"], value="MONTH", label="薪資類型")
        create_btn = gr.Button("新增", variant="primary")
        create_btn.click(
            _create,
            [name, age, phone, identity, salary_type],
            employees_table,
        )

    with gr.Accordion("刪除員工", open=False):
        delete_id = gr.Number(label="員工 ID", precision=0)
        delete_btn = gr.Button("刪除", variant="stop")
        delete_btn.click(_delete, delete_id, employees_table)

    with gr.Accordion("管理請假 / 不可用日期", open=False):
        emp_id = gr.Number(label="員工 ID", precision=0)
        unavail_table = gr.Dataframe(
            headers=["id", "type", "day_of_week", "start_date", "end_date", "reason"],
            interactive=False,
        )
        list_btn = gr.Button("查詢")
        list_btn.click(_list_unavail, emp_id, unavail_table)

        with gr.Row():
            unavail_type = gr.Dropdown(
                ["DAY_OF_WEEK", "DATE_RANGE"], value="DAY_OF_WEEK", label="類型"
            )
            day_of_week = gr.Number(
                label="星期幾 (1=Mon..7=Sun)", precision=0, minimum=1, maximum=7
            )
            start_date = gr.Textbox(label="開始日期 (YYYY-MM-DD)")
            end_date = gr.Textbox(label="結束日期 (YYYY-MM-DD)")
            reason = gr.Textbox(label="原因")
        add_btn = gr.Button("新增不可用日期", variant="primary")
        add_btn.click(
            _add_unavail,
            [emp_id, unavail_type, day_of_week, start_date, end_date, reason],
            unavail_table,
        )
