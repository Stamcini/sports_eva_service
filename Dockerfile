FROM python:3.9

# 设置 python 环境变量
ENV PYTHONUNBUFFERED 1

# 设置pip源为国内源
COPY pip.conf /root/.pip/pip.conf

#COPY sources.list /etc/apt/sources.list
#RUN apt update
#RUN apt install libgl1-mesa-glx

# 在容器内创建文件夹
RUN mkdir -p /var/www/html/sports_eva_service

# 设置容器内工作目录
WORKDIR /var/www/html/sports_eva_service

COPY requirements.txt ./requirements.txt

# pip安装依赖
RUN pip install -r requirements.txt

EXPOSE 8000

# RUN export PYTHONPATH="/var/www/html/sports_eva_service:$PYTHONPATH"
ENV PYTHONPATH = "/var/www/html/sports_eva_service:$PYTHONPATH"

# 将当前目录文件加入到容器工作目录中（. 表示当前宿主机目录）
ADD . /var/www/html/sports_eva_service

CMD python3 ./the_server/manage.py runserver 0.0.0.0:8000

