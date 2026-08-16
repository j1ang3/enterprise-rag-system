FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Keep dependency installation in a separate layer so source-only changes can
# reuse it, and avoid retaining pip's download cache in the runtime image.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy only the files needed to run the API and invoke migrations explicitly.
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini

# Application imports create the storage tree, while structured logging creates
# its parent lazily. Preparing both here gives the non-root process safe write
# access without copying local documents, indexes, or logs into the image.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p storage/uploads storage/texts storage/index logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
