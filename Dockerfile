FROM ubuntu:16.04
FROM python:3.7


COPY requirements.txt /

COPY Flowers.ipynb /
COPY model.h5 /
COPY test /test/test1

COPY weights_flowers.hdf5 /

RUN pip install -r requirements.txt

ADD inference.py /
RUN chmod u+x inference.py

CMD [ "python", "./inference.py" ]
