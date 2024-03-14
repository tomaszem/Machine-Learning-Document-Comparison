# Machine Learning Document Comparison
## Running server

Nejdříve

bash
`pip install chromadb`

Start server

bash
`chroma run --path /db_path`

Pro perzistentní úložiště

bash
`client = chromadb.PersistentClient(path="/path/to/save/to")`
