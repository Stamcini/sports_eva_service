FROM python:3.9

# 设置 python 环境变量
ENV PYTHONUNBUFFERED 1

# set the sources
COPY pip.conf /root/.pip/pip.conf
RUN pip install --upgrade pip

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
COPY sources.list /etc/apt/

# solve the problem of libgl.so.1 not found
RUN apt-get update && apt-get install libgl1-mesa-glx -y

# 在容器内创建文件夹
RUN mkdir -p /var/www/html/sports_eva_service

# 设置容器内工作目录
WORKDIR /var/www/html/sports_eva_service

COPY requirements.txt ./requirements.txt

# pip安装依赖
RUN pip install -r requirements.txt

EXPOSE 8000

# RUN export PYTHONPATH="/var/www/html/sports_eva_service:$PYTHONPATH"
ENV PYTHONPATH "/var/www/html/sports_eva_service:$PYTHONPATH"

# 将当前目录文件加入到容器工作目录中（. 表示当前宿主机目录）
RUN mkdir -p /var/www/html/sports_eva_service/the_server/temp

COPY ./the_server/server_django /var/www/html/sports_eva_service/the_server/server_django/
COPY ./the_server/utils /var/www/html/sports_eva_service/the_server/utils/
COPY ./the_server/static /var/www/html/sports_eva_service/the_server/static/
COPY ./the_server/configs /var/www/html/sports_eva_service/the_server/configs/

COPY ./the_server/manage.py ./the_server/db.sqlite3 /var/www/html/sports_eva_service/the_server/


CMD python3 ./the_server/manage.py runserver 0.0.0.0:8000

