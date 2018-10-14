FROM tensorflow/tensorflow:latest-devel-gpu-py3

RUN apt update && \
    apt-get install -y libsm6 libxext6 && \
    apt-get install -y libfontconfig1 libxrender1

RUN pip install Cython && \
    pip install h5py && \
    pip install imgaug && \
    pip install Keras && \
    pip install matplotlib && \
    pip install numpy && \
    pip install opencv-python && \
    pip install Pillow && \
    pip install scikit-image && \
    pip install scipy && \
    pip install pandas && \
    pip install tqdm && \
    pip install pydicom

RUN git clone --depth 1 https://github.com/khyu777/rsna_model.git /rsna_model
RUN cd /rsna_model && git submodule update --init
ENV PYTHONPATH $PYTHONPATH:/rsna_model/Mask_RCNN
WORKDIR /rsna_model
