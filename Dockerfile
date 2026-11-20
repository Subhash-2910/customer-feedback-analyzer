FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
# CPU-only PyTorch keeps the image ~2 GB smaller than the CUDA build.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
# The fine-tuned model is not in git (478 MB), so build it into the image: downloads GPT-2 once and
# fine-tunes on CPU (a few minutes).
RUN python -m app.dataset && python -m app.train
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "streamlit run app/main.py --server.address 0.0.0.0 --server.port ${PORT}"]
