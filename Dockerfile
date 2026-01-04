# Set the base image (environment) to create the image for our RAG system application
FROM python:3.13.2-slim-bookworm

# Install uv inside the docker container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the work directory inside the container
WORKDIR /app

# Copy the dependencies inside the container
COPY pyproject.toml uv.lock ./

# Install the dependencies inside the container
RUN uv sync --frozen

# Copy all the files from the current local root directory to the /app directory inside the container
COPY . .

# Expose the port
EXPOSE 7860

# Run the command to start the chatbot using the virtual environment we created
CMD [ "uv", "run", "-m", "core.src.ui.app" ]