FROM gcr.io/kubeflow-images-public/tensorflow-1.13.1-notebook-gpu:v0.5.0
COPY requirements.txt /tmp/
ADD src /app
RUN pip install --requirement /tmp/requirements.txt
EXPOSE 80
#CMD ["python", "/app/test_gpu.py"]

#run using interactive model 
#build the image use docker file or check out from dockerhub
#docker run -p 8000:40 YourDockerImageName -it







