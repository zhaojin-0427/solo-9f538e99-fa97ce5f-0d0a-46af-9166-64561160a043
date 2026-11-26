FROM python:3.12-slim

WORKDIR /srv/app

COPY app/requirements.txt /srv/app/requirements.txt
RUN pip install --no-cache-dir -r /srv/app/requirements.txt

COPY app/ /srv/app/
COPY samples/ /srv/samples/

ENV SAMPLES_DIR=/srv/samples \
    PORT=8000
EXPOSE 8000

# waitress：多线程 WSGI 服务器。某个下载连接停读/断连只占用其自身线程，
# 不影响其他窗口的保存与审听。
CMD ["sh", "-c", "waitress-serve --listen=0.0.0.0:${PORT} app:app"]
