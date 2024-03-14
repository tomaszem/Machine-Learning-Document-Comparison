# Machine Learning Document Comparison
##Running server
Nejdříve
bash
`pip install chromadb`
Start server
`chroma run --path /db_path`
Pro perzistetní úložiště
bash
`client = chromadb.PersistentClient(path="/path/to/save/to")`
