FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN mkdir /dy
WORKDIR /dy

ENV AA=asd

RUN apt update \
&& apt install -y python3 \
&& apt install -y python3-pip git \
&& pip3 install jupyter tqdm pillow pyyaml scipy \
&& apt install -y libsm6 libxext6 libxrender1 libfontconfig1 \
&& pip3 install opencv-python opencv-contrib-python 

RUN pip3 install tensorflow-gpu==1.14.0

ENTRYPOINT ["/bin/sh", "-c", "jupyter-notebook --no-browser --ip=0.0.0.0 --NotebookApp.token='kdy' --allow-root --NotebookApp.password='' --NotebookApp.allow_origin='*'"]

