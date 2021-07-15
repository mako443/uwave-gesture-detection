FROM alpine

RUN apk add alpine-sdk

#Install python
RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools

# Install packages
RUN apk add python3-dev
RUN pip3 install wheel
RUN pip3 install uvicorn
RUN pip3 install uvloop
RUN pip3 install requests
RUN pip3 install aiofiles
RUN pip3 install fastapi
RUN pip3 install numpy
RUN pip3 install python-multipart
RUN pip3 install jinja2

#Workdir and files
WORKDIR /app/
COPY ./api ./api
COPY ./checkpoints ./checkpoints
COPY ./dataloading ./dataloading
COPY ./models ./models

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "--port", "80", "api.main:app"]