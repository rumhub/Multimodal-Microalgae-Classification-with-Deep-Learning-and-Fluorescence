FROM tensorflow/tensorflow:2.9.2-gpu-jupyter

USER root

# --------------------------
# 🔧 Dependencias del sistema
# --------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# 🧩 Librerías científicas
# --------------------------
RUN pip install --no-cache-dir \
    opencv-python==4.12.0.88 \
    SimpleITK==2.3.1 \
    numpy==1.23.2 \
    tqdm==4.67.1 \
    matplotlib==3.7.1

# --------------------------
# 🧠 JupyterLab + autocompletado
# --------------------------
RUN pip install --no-cache-dir \
    jupyterlab==3.6.7 \
    ipython==8.12.0 \
    jedi==0.18.1 \
    parso==0.8.3 \
    jupyterlab-lsp==4.3.0 \
    python-lsp-server[all]==1.10.0

# ⚙️ Actualizar componentes base de Jupyter para evitar el error "ensure_async"
RUN pip install --no-cache-dir --upgrade \
    jupyter_core==5.3.0 \
    jupyter_server==1.24.0

# --------------------------
# 📁 Directorio de trabajo
# --------------------------
WORKDIR /tf
RUN mkdir -p /tf/codigo && chmod -R 777 /tf

# --------------------------
# 🚀 Ejecutar como root
# --------------------------
USER root

# --------------------------
# 🌐 Exponer puerto de Jupyter
# --------------------------
EXPOSE 8888

# --------------------------
# 🧭 Comando de inicio
# --------------------------
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token="]