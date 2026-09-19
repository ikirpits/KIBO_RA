import modal

app = modal.App("kibo-ra")
image = modal.Image.from_dockerfile("Dockerfile").workdir("/app")


@app.function(image=image, memory=4096, timeout=600, min_containers=0)
@modal.asgi_app()
def web():
    from app import app as fastapi_app
    return fastapi_app
