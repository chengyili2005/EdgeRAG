# RAG
Implementing a RAG system


Setup:
0. Setup a conda environment
```bash
conda create -n EdgeRAG python=3.11
conda activate EdgeRAG
pip install requirements.txt
pip install ipykernel # if working on the notebook
```

1. Install postgresql
```bash
sudo apt install postgresql
```

2. Install pgvector
```bash
sudo apt install postgresql-{$postgres_server_version}-pgvector
```
  - Where `postgres_server_version` should be what server version you see when you do:
```bash
sudo -u postgres psql
SELECT version();
```

2. Set up postgresql
```bash
# First open postgres with the default username
sudo -u postgres psql

# Then create a role for yourself
CREATE ROLE username WITH LOGIN SUPERUSER CREATEDB CREATEROLE PASSWORD 'password';

# Exit
\q
```

3. Create a database
```bash

# (Bash) create a database
PGPASSWORD='password' createdb -U username -h localhost -p 5432 edgerag_db

# (Bash) go into the database
PGPASSWORD='password' psql -U username -h localhost -p 5432 edgerag_db

# (PSQL) enable pgvector extension
CREATE EXTENSION vector;
```

Should be good to go...