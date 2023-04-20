FROM ubuntu

RUN apt-get update && apt-get install -y python3-dev python3-pip libfreetype6-dev libpng-dev libhdf5-dev

RUN pip3 install --upgrade pip

RUN mkdir /app

ADD . /app

COPY requirements1.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["/bin/bash"]