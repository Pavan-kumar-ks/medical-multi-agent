API-mode Docker image

This repository includes `Dockerfile.api` and `requirements-api.txt` for running the app
without installing heavy local ML libraries. In this mode the application calls the
Hugging Face Inference API to compute embeddings instead of loading `sentence-transformers`.

Build and run (example):

1. Build the image:

   docker build -f Dockerfile.api -t medical-agent:api .

2. Run the container (set your HF token):

   docker run -e HUGGINGFACE_HUB_TOKEN=hf_xxx -p 8000:8000 medical-agent:api

Notes:
- The container uses `MED_AGENT_USE_HF_API=1` by default; the app will call the HF inference API
  for embeddings when `HUGGINGFACE_HUB_TOKEN` is provided.
- Use the lighter image when you do not want heavy native dependencies on your host.
