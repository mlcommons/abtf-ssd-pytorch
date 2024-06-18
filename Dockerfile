FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN pip install Cython
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install faster-coco-eval
RUN pip install torchinfo
#RUN pip install typing-extensions
RUN pip install torchmetrics
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

COPY . .
