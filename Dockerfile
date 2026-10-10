FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ARG PDFIDELITY_ENGINE_REV=e6770a89ee8e7319ab710a57ea9cdc5881ca4e34
LABEL org.opencontainers.image.source="https://github.com/andygeek/pdfidelity-translator"
LABEL org.opencontainers.image.title="PDFidelity Translator"
LABEL com.pdfidelity.engine.revision="${PDFIDELITY_ENGINE_REV}"

WORKDIR /app


EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# # Download all required fonts
# ADD "https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf" /app/
# ADD "https://github.com/timelic/source-han-serif/releases/download/main/SourceHanSerifCN-Regular.ttf" /app/
# ADD "https://github.com/timelic/source-han-serif/releases/download/main/SourceHanSerifTW-Regular.ttf" /app/
# ADD "https://github.com/timelic/source-han-serif/releases/download/main/SourceHanSerifJP-Regular.ttf" /app/
# ADD "https://github.com/timelic/source-han-serif/releases/download/main/SourceHanSerifKR-Regular.ttf" /app/

RUN apt-get update && \
     apt-get install --no-install-recommends -y libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 build-essential && \
     rm -rf /var/lib/apt/lists/*

COPY . .

# Install the selected PDFidelity engine instead of upgrading to upstream PyPI.
RUN uv pip install --system --no-cache --compile-bytecode \
     "BabelDOC @ https://github.com/andygeek/pdfidelity-engine/archive/${PDFIDELITY_ENGINE_REV}.tar.gz" . && \
     babeldoc --version && babeldoc --warmup
RUN pdf2zh --version
CMD ["pdf2zh", "--gui"]
