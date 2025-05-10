
# slim bookworm has no gpu support
#FROM python:3.12.7-slim-bookworm
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime  
#PyTorch CUDA image (ready for GPU)


# The installer requires curl (and certificates) to download the release archive
# build-essential is for "MAKE"

#RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential ca-certificates
# Install system dependencies
# #============================================
# RUN apt-get update -y && apt install -y --no-install-recommends git\
#     && pip install --no-cache-dir -U pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates procps lsb-release \
    && rm -rf /var/lib/apt/lists/*
#========================================

# Install uv package manager this part is for UV so commenting out ===============
# Download the latest installer
# ADD https://astral.sh/uv/install.sh /uv-installer.sh

# # Run the installer then remove it
# RUN sh /uv-installer.sh && rm /uv-installer.sh

# # Ensure the installed binary is on the `PATH`
# ENV PATH="/root/.local/bin/:$PATH"

# WORKDIR /mlops

# Option 2: If you're using pyproject.toml with UV
# copy_ONLY the pyproject.toml and other build files first


# COPY pyproject.toml setup.py setup.cfg ./
# Install the package in development mode

# # Install Python dependencies from pyproject.toml
# RUN uv pip install --system .


# COPY . .



#=========================================== end uv part==============

# Set working directory
WORKDIR /mlops 

# Copy requirements file
COPY requirements.cpu.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.cpu.txt

# Copy application code
COPY . .


#===========================

# Verify key packages are installed
RUN python -c "import lightning; print(f'Lightning version: {lightning.__version__}')"
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import dvc; print(f'DVC version: {dvc.__version__}')"

#===================================

CMD [ "bash" ]
