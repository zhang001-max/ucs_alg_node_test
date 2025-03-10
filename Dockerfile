FROM python:3.8-slim
LABEL authors="zhangwenwen"
ADD ./ app
WORKDIR /app
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install ./libs/ucs_alg_node-0.1.5-py3-none-any.whl
RUN pip3 install -r requirements.txt
COPY . .

ENTRYPOINT ["python3", "demo.py"]
