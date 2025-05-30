# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.10    (apt)
# pytorch       latest (pip)
# tensorflow    latest (pip)
# keras         latest (pip)
# opencv        4.1.2  (git)
# sonnet        latest (pip)
# caffe         latest (git)
# ==================================================================

FROM nvcr.io/nvidia/pytorch:24.01-py3
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar &&\

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j24 install && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        && \
    wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py &&\
    python3.10 ~/get-pip.py && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn \
        matplotlib \
        Cython \
	setuptools>=41.0.0 \
        && \

# ==================================================================
# boost
# ------------------------------------------------------------------

    #wget -O ~/boost.tar.gz https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz && \
    #tar -zxf ~/boost.tar.gz -C ~ && \
    #cd ~/boost_* && \
    #./bootstrap.sh --with-python=python3.7 && \
    #./b2 install -j"$(nproc)" --prefix=/usr/local && \

# ==================================================================
# fix for tf
# ------------------------------------------------------------------

    #pip install wrapt --upgrade --ignore-installed && \

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

   #pip install wrapt==1.10.0 --ignore-installed && \

   # $PIP_INSTALL \
   #     tf-nightly \
   #     && \

# ==================================================================
# keras
# ------------------------------------------------------------------

   # $PIP_INSTALL \
   #     h5py \
   #     keras \
   #     && \

# ==================================================================
# opencv
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        && \

    $GIT_CLONE --branch 4.x https://github.com/opencv/opencv ~/opencv && \
    mkdir -p ~/opencv/build && cd ~/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_EXAMPLES=OFF \
          .. && \
    make -j24 install && \
    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2 && \

# ==================================================================
# sonnet
# ------------------------------------------------------------------

   # $PIP_INSTALL \
   #     tensorflow_probability \
   #     dm-sonnet \
   #     && \

# ==================================================================
# caffe
# ------------------------------------------------------------------

   # apt-get update && \
   # DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
   #     caffe-cuda \
   #     && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
# ==================================================================
# Installing APEX
# ------------------------------------------------------------------

#WORKDIR /tmp/apex-dir
#RUN pip uninstall -y apex || :
#RUN pip uninstall -y apex || :
#RUN SHA=ToUcHmE git clone https://github.com/NVIDIA/apex.git
#WORKDIR /tmp/apex-dir/apex
#RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
#WORKDIR /workspace
# ------------------------------------------------------------------
# Jupyter stuff
# ------------------------------------------------------------------
RUN pip install jupyterlab
RUN jupyter notebook --generate-config
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN echo "c.NotebookApp.password = 'sha1:2e471d7efc16:054bb90793eb5c1cdf64d427c4538a710469e2fd'" > /root/.jupyter/jupyter_notebook_config.py
#------compats for BioNLP
#RUN echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list
#RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
#RUN apt-get update
#RUN apt-get -y install postgresql
#RUN apt-get -y install postgresql-server-dev-all

#RUN pip install ufal.udpipe
#RUN pip install tika 
#RUN pip install textract 
#RUN pip install git+https://github.com/IINemo/isanlp.git 
#RUN pip install psycopg2 

# ------------------------------------------------------------------
# Two folders, probably for different storages
# ------------------------------------------------------------------
RUN mkdir /workspace/notebooks && mkdir /workspace/data



#COPY postgresql.conf /etc/postgresql/13/main/postgresql.conf
#COPY pg_hba.conf /etc/postgresql/13/main/pg_hba.conf
#COPY protdb.tar.gz /workspace/notebooks

#USER postgres

#RUN    /etc/init.d/postgresql start &&\
#    psql --command "CREATE USER docker WITH SUPERUSER PASSWORD 'pcydonia32';" &&\
#    createdb -O docker prot_main_01 &&\
#    pg_restore -F c -d prot_main_01 /workspace/notebooks/protdb.tar.gz

#USER root

EXPOSE 2408 8888

CMD jupyter-lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir='/workspace'
