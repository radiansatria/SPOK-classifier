ROM heroku/miniconda

# Grab requirements.txt.
ADD /requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# Add our code
ADD ./ /opt/webapp/
WORKDIR /opt/webapp



CMD gunicorn --bind 0.0.0.0:$PORT run_keras_server:app
