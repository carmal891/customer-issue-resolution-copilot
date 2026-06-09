# Docker Deployment

This directory contains Docker configuration files for containerized deployment of the Customer Issue Resolution Copilot.

## Files

- **Dockerfile** - Container image definition
- **docker-compose.yml** - Multi-container orchestration
- **.dockerignore** - Files to exclude from Docker build context
- **docker-entrypoint.sh** - Container startup script with database cleanup

## Quick Start

### Prerequisites

- Docker Desktop installed and running
- OpenAI API key (see Configuration section below)

### Running the Application

From the **project root directory**, run:

```bash
cd ci
docker-compose up --build
```

Or use the shorthand from project root:

```bash
docker-compose -f ci/docker-compose.yml up --build
```

### Accessing the Application

Once the container is running, open your browser to:

- **URL:** http://localhost:8501

### Stopping the Application

```bash
cd ci
docker-compose down
```

## Configuration

### Environment Variables - OpenAI API Key

There are **three ways** to pass the OpenAI API key to Docker Compose:

#### Method 1: Using .env file (Recommended)

Create a `.env` file in the **project root directory** (not in ci/):

```bash
# In project root directory
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-actual-api-key-here
EOF
```

Then run docker-compose:
```bash
cd ci
docker-compose up --build
```

Docker Compose automatically reads the `.env` file from the parent directory.

#### Method 2: Export as environment variable

```bash
# Export in your shell
export OPENAI_API_KEY=sk-your-actual-api-key-here

# Then run docker-compose
cd ci
docker-compose up --build
```

#### Method 3: Inline with docker-compose command

```bash
cd ci
OPENAI_API_KEY=sk-your-actual-api-key-here docker-compose up --build
```

### Optional Environment Variables

You can override any configuration in your `.env` file:

```env
# Required
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional - Override defaults
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
CHUNK_SIZE=512
TOP_K_RETRIEVAL=10
LOG_LEVEL=INFO
```

See `.env.example` in the project root for all available options.

### Volume Mounts

The following directories are mounted from the host:

- `../data/skills` - Skill definitions (YAML files)
- `../data/policies` - Policy documents
- `../data/vector_db` - ChromaDB persistent storage
- `../logs` - Application logs

## Troubleshooting

### ChromaDB Schema Error

If you see `no such column: collections.topic` error:

```bash
# Stop containers and remove volumes
cd ci
docker-compose down -v

# Clean local database
rm -rf ../data/vector_db
mkdir -p ../data/vector_db

# Rebuild without cache
docker-compose build --no-cache
docker-compose up
```

### Container Won't Start

Check logs:
```bash
cd ci
docker-compose logs -f app
```

### Port Already in Use

If port 8501 is already in use, modify `docker-compose.yml`:

```yaml
ports:
  - "8502:8501"  # Change host port to 8502
```

### Missing API Key Error

If you see "OpenAI API key not found" error:

1. Verify your `.env` file exists in the project root (not in ci/)
2. Check that `OPENAI_API_KEY` is set correctly
3. Try Method 2 or Method 3 from the Configuration section above

## Development

### Rebuilding After Code Changes

```bash
cd ci
docker-compose down
docker-compose up --build
```

### Accessing Container Shell

```bash
docker exec -it customer-copilot-app bash
```

### Viewing Logs

```bash
cd ci
docker-compose logs -f app
```

## Production Considerations

For production deployment:

1. **Use secrets management** - Don't use `.env` files
2. **Add authentication** - Implement proper user authentication and authorization
3. **Set up reverse proxy** - Use nginx or similar
4. **Enable HTTPS** - Use SSL certificates
5. **Configure resource limits** - Set memory and CPU limits in docker-compose.yml
6. **Set up monitoring** - Add health checks and logging aggregation
7. **Use external vector DB** - Consider hosted ChromaDB or Pinecone

## Architecture

The Docker setup uses:

- **Single container** - Streamlit app with embedded ChromaDB
- **Local ChromaDB mode** - PersistentClient for simplicity
- **Volume mounts** - For data persistence and easy updates
- **Health checks** - Automatic container restart on failure
- **Entrypoint script** - Automatic database cleanup on startup