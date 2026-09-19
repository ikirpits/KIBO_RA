import modal

app = modal.App("kibo-ra-ui")
image = modal.Image.from_dockerfile("Dockerfile").workdir("/app")


@app.function(image=image, memory=4096, timeout=600, min_containers=0)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from ui import demo

    return gr.mount_gradio_app(FastAPI(), demo, path="/")
