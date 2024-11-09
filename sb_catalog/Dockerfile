# set base image (host OS)
FROM python:3.12

# set the working directory in the container
WORKDIR /code

# install dependencies
# pytorch CPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# SeisBench
RUN pip install seisbench pyocto ipython s3fs pymongo boto3 tqdm

# Get AWS DocumentDB public key
RUN wget https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

# copy the content of the local src directory to the working directory
COPY src/ src/

# command to run on container start
ENTRYPOINT ["python", "-m", "src.picker"]
