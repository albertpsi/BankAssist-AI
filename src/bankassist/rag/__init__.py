"""Basic Retrieval-Augmented Generation over the banking policy corpus (Lab 2).

Documents → chunks → embeddings → Pinecone → top-k similarity → grounded answer.

Deliberately basic: plain vector similarity, no hybrid search, no filtering, no
reranking, no query rewriting. Those are Lab 3 and are added alongside this
pipeline rather than inside it, so the two can be compared.
"""
