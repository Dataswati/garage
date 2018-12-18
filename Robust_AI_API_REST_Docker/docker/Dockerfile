FROM ubuntu:16.04


MAINTAINER Sebastien Guissart "sguissart@dataswati.com"


RUN apt -y update &&\
    apt -y install python3 python3-pip

RUN python3 -m pip install --upgrade pip


RUN apt -y update &&\
    apt install -y nginx supervisor

# install redis q
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    redis-server

#install all python modules
ADD ./src/requirements.txt /
RUN pip3 install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# entrypoint
COPY /docker-entrypoint.sh /
COPY /docker-entrypoint.d/* /docker-entrypoint.d/
ONBUILD COPY /docker-entrypoint.d/* /docker-entrypoint.d/

RUN chmod +x docker-entrypoint.sh
RUN chmod +x docker-entrypoint.d/*.sh

# nginx 
RUN rm /etc/nginx/sites-enabled/default
ADD ./nginx_flask_proxy.conf /etc/nginx/sites-enabled/
RUN echo "daemon off;" >> /etc/nginx/nginx.conf

# supervisor log
RUN mkdir -p /var/log/supervisor

ENTRYPOINT ["/usr/bin/supervisord"]
# ENTRYPOINT ["/docker-entrypoint.sh"]

# RUN pip3 install gunicorn
#     pip3 install rq

