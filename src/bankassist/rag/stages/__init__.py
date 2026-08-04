"""Enterprise RAG pipeline stages (Lab 3).

Each stage is an independent class with exactly one public method, ``execute``,
taking and returning typed objects — never a ``dict``. Stages are constructed
with their collaborators injected, so each is unit-testable without the rest of
the pipeline. ``pipeline.enterprise_pipeline.EnterpriseRagPipeline`` is the only
module that sequences them.
"""
