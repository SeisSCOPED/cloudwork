# set base image (host OS)
FROM python:3.11

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
# COPY requirements.txt .

# install dependencies
# pytorch CPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# SeisBench
RUN pip install seisbench
# Pyocto
RUN pip install pyocto
# IPython
RUN pip install ipython
# S3FS
RUN pip install s3fs
# PyMongo
RUN pip install pymongo

# copy the content of the local src directory to the working directory
COPY src/ .

# command to run on container start
ENTRYPOINT ["python", "picker.py"]
