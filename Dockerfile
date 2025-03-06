FROM python:3.12

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies

RUN pip install -v --no-cache-dir --default-timeout=500 -r requirements.txt

# Copy your script
COPY xgboost_sol.py .


# Command to run your script
COPY entrypoint.sh .
ENTRYPOINT ["./entrypoint.sh"]
CMD ["/bin/bash"]
