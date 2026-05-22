# reranker

- CUDA 13.0

```bash
pip install -r requirements.txt
```

- CUDA 12.9

```bash
pip install -r requirements.txt \
    --extra-index-url https://wheels.vllm.ai/0.21.0/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
```
