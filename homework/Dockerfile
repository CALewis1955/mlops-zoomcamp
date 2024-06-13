FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here

# upgrade pip
RUN pip install --upgrade pip

# install pipenv
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["starter.py", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our app to be running
ENTRYPOINT ["./starter.py",  "--year", "2023", "--month", "05"]