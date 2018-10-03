# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:nightly-py3

# Set the working directory to /app
WORKDIR /mlw

# Copy the current directory contents into the container at /app
ADD . /mlw

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN apt-get -y update
RUN apt-get -y install vim 
RUN apt-get -y install nano

# Make port 8888 available to the world outside this container
EXPOSE 8888 6006 6064

# Define environment variable
ENV NAME mlw

# Run jupyter notebook when the container launches
CMD ["/mlw/start.sh"]
