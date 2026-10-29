"""The web app on Modal, built from the same Dockerfile as any other host.

modal deploy deploy/modal_app.py    (from the repository root)
"""

import modal

image = modal.Image.from_dockerfile("Dockerfile")
data = modal.Volume.from_name("pollicount-data", create_if_missing=True)
app = modal.App("pollicount", image=image)


@app.function(
    secrets=[modal.Secret.from_name("pollicount-keys")],
    volumes={"/data": data},
    max_containers=1,  # analyses are tracked in memory, so every request must reach the same container
    scaledown_window=15 * 60,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from pollicount.server import production_app

    return production_app()
