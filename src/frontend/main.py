import gradio as gr

from frontend.views import ai_scheduling, employees, policies, schedules


def build_app() -> gr.Blocks:
    with gr.Blocks(title="班表查詢系統", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 班表查詢系統")
        with gr.Tabs():
            with gr.Tab("班表查詢"):
                schedules.render()
            with gr.Tab("AI 排班"):
                ai_scheduling.render()
            with gr.Tab("員工管理"):
                employees.render()
            with gr.Tab("排班政策"):
                policies.render()
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
