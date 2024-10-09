# This base image provides Tensorflov v2.7.0 with support for both Intel and ARM 
FROM sarusso/tensorflow:2.7.0

COPY requirements_pinned.txt /tmp/requirements_pinned.txt
RUN pip3 install -r /tmp/requirements_pinned.txt
