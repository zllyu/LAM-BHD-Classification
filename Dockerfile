FROM nvcr.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG PYTHON_VERSION=3.12.10
ARG PYTHON_MAJOR_MINOR=3.12
ARG PIP_VERSION=25.3
ARG SETUPTOOLS_VERSION=80.1.0
ARG WHEEL_VERSION=0.45.1
ARG UID=5642
ARG GID=5642

ENV LC_ALL="C.UTF-8" \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LAM_BHD_DATA_DIR=auto

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libbz2-dev \
        libffi-dev \
        libgdbm-compat-dev \
        libgdbm-dev \
        liblzma-dev \
        libncursesw5-dev \
        libnss3-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        llvm \
        tk-dev \
        wget \
        xz-utils \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o Python.tgz && \
    tar -xzf Python.tgz && \
    cd "Python-${PYTHON_VERSION}" && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make install && \
    cd .. && \
    rm -rf "Python-${PYTHON_VERSION}" Python.tgz

RUN ln -sf "/usr/local/bin/python${PYTHON_MAJOR_MINOR}" /usr/bin/python3 && \
    ln -sf "/usr/local/bin/pip${PYTHON_MAJOR_MINOR}" /usr/bin/pip3

RUN <<"EOF" bash
    set -eu -o pipefail
    if [[ $GID -ge 1000  ]] && getent group $GID >/dev/null; then
        group_name="$(getent group $GID | cut -d: -f1)"
        groupdel "$group_name" >/dev/null
    fi
    getent group $GID >/dev/null || groupadd -r -g $GID appgroup
    if [[ $UID -ne 0 ]]; then
        if [[ $UID -ge 1000 ]] && getent passwd $UID >/dev/null; then
            user_name="$(getent passwd $UID | cut -d: -f1)"
            userdel "$user_name" >/dev/null
        fi
        useradd -m -d /workspace -l -s /bin/bash -g $GID -N -u $UID appuser
    fi
EOF

ENV VIRTUAL_ENV="/venv"
RUN python3 -m venv "$VIRTUAL_ENV" && \
    "$VIRTUAL_ENV/bin/python" -m pip install --no-cache-dir --upgrade \
        "pip==${PIP_VERSION}" \
        "setuptools==${SETUPTOOLS_VERSION}" \
        "wheel==${WHEEL_VERSION}" && \
    chown -R $UID:$GID "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_ROOT_USER_ACTION=ignore pip install --requirement /requirements.txt && \
    chown -R $UID:$GID "$VIRTUAL_ENV"

RUN mkdir -p /workspace && \
    chown $UID:$GID /workspace

WORKDIR /workspace
USER $UID

COPY --chown=$UID:$GID ./app ./app
COPY --chown=$UID:$GID ./meta.json ./

ENV PYTHONPATH="/workspace/app/custom"
