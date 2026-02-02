FROM nvidia/cuda:12.2.0-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

#For python3.9
RUN apt-get update && apt-get install --no-install-recommends -y python3.9 python3.9-dev python3.9-venv python3-pip python3-wheel build-essential && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

# For python3.1
#RUN apt update
#RUN apt-get install software-properties-common -y
#RUN add-apt-repository ppa:deadsnakes/ppa -y
#RUN apt update
#RUN apt upgrade -y
#RUN apt-get update && apt-get install --no-install-recommends -y python3.10 python3.10-venv python3-pip python3-wheel build-essential && \
#	apt-get clean && rm -rf /var/lib/apt/lists/*

# create and activate virtual environment
# using final folder name to avoid path issues with packages
RUN python3.9 -m venv /home/myuser/venv
#RUN python3.10 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

RUN pip3 install --no-cache-dir wheel

RUN apt-get update && apt-get install -y cuda-toolkit-12-2

#RUN apt update
#RUN apt-get install -y python3.9 python3-pip

#Comentei tudo abaixo
RUN pip install jupyterlab==4.1.6
#RUN pip install pandas==2.2.2
#RUN pip install scikit-learn==1.4.2
#RUN pip install matplotlib==3.8.4
#RUN pip install seaborn==0.13.2

#RUN pip install transformers==4.37.2
#RUN pip install tensorflow==2.15.0
#RUN pip install torch==2.3.0 
#RUN pip install torchvision==0.18.0
#RUN pip install torchaudio==2.3.0
#RUN pip install unidecode==1.3.8
#RUN pip install scipy==1.10.1
#RUN pip install nltk==3.8.1
#RUN pip install gensim==4.3.2
#RUN pip install spacy==3.7.4
#RUN pip install yake==0.4.8
#RUN pip install ipywidgets==8.1.2

#RUN pip install langchain-community==0.0.34
#RUN pip install langchain==0.1.17
#RUN pip install keybert==0.8.5
#RUN pip install scikit-optimize==0.10.2
#RUN pip install wordcloud==1.9.3
#RUN pip install plotly-express==0.4.1
#RUN pip install groq==0.31.0
#RUN pip install textstat==0.7.10
#RUN pip install skfeature-chappers==1.1.0

#comentei tudo acima

#TPOT
#RUN pip install tpot

#AutoGluon-Text
#RUN apt-get update && apt-get install -y \
#    build-essential \
#    cmake \
#    pkg-config \
#    libssl-dev \
#    libffi-dev \
#    python3-dev \
#    libbz2-dev \
#    liblzma-dev \
#    zlib1g-dev \
#    libcurl4-openssl-dev \
#    libsnappy-dev \
#    libbrotli-dev \
#    libzstd-dev \
#    libboost-filesystem-dev \
#    libboost-system-dev \
#    libboost-regex-dev \
#    libboost-thread-dev \
#    && rm -rf /var/lib/apt/lists/*

#RUN pip install --upgrade pip setuptools wheel

#RUN pip install \
#    jupyterlab==3.6.7 \
#    jupyter-server==1.24.0 \
#    jupyter-events==0.6.3 \
#    jsonschema==4.17.3 \
#    referencing==0.30.2

#RUN pip install \
#    transformers==4.30.2 \
#    tensorflow==2.12.1 \
#    protobuf==3.20.3 \
#    torch==2.0.1 \
#    torchvision==0.15.2 \
#    torchaudio==2.0.2

#RUN pip install autogluon==0.8.2
#RUN pip install autogluon.text==0.6.3b20230123

#AutoViML
#RUN pip install autoviml

#Auto-sklearn

#RUN pip install "dask==2022.12.1" "distributed==2022.12.1"
#RUN pip install scipy==1.8.1
#RUN pip install Cython==0.29.35
#RUN pip install scikit-learn==0.24.2 --no-build-isolation
#RUN pip install auto-sklearn

#RUN pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# Set the working directory inside the container
WORKDIR /app

# Copia TODO o projeto (inclusive pyproject.toml)
COPY pyproject.toml /app

# Instala o pacote autonlp no modo desenvolvimento
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e .

COPY text2meta /app/text2meta

# Expose the Jupyter port
EXPOSE 8888

# Command to run Jupyter Notebook
#CMD [ "/bin/bash" ]
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
