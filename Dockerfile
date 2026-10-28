# CUDA 12.8 / torch 2.7 supports the local Blackwell GPU.
# Pin the resolved registry digest in image-lock.json before execution.
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG SOURCE_HASH=unrecorded
LABEL org.opencontainers.image.title="WinoBias reproduction smoke" \
      org.opencontainers.image.revision="${SOURCE_HASH}"
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
COPY requirements.txt /app/requirements.txt
COPY wheelhouse/ /opt/wheelhouse/
RUN python -m pip install --no-cache-dir --no-index --find-links=/opt/wheelhouse -r /app/requirements.txt \
    && python -m pip check \
    && python -m pip freeze --all > /app/environment-freeze.txt
COPY wheelhouse-manifest.json /app/wheelhouse-manifest.json
COPY prepare.py protocol.py parser_v3.py parser_v4.py resume_journal.py run.py /app/
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--manifest", "/app/configs/smoke.json", "--prepare-only"]
