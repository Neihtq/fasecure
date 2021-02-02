FROM tiangolo/uwsgi-nginx-flask:python3.8
COPY . /backend
RUN apt-get update
RUN apt-get install -y --reinstall ca-certificates
RUN pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.htmlnumpy pandas flask requests matplotlib Pillow pytorch-lightning scikit-learn opencv-python pytorch-metric-learning pickle5
CMD [ "python", "/backend/server.py"]
