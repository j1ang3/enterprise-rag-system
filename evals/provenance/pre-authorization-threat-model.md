# Historical Pre-Authorization Threat Boundary

This public provenance note represents the threat-model input used by the frozen prompt-injection experiments before JWT authentication, document ownership, ACLs, and permission-aware retrieval were implemented.

The historical experiment modeled a reachable shared-corpus RAG service and focused on direct prompt injection, indirect prompt injection, prompt leakage, malicious-document persistence, context poisoning, citation manipulation, and logging exposure.

This file is an experiment identity only. The current system is documented by `docs/security/threat-model.md` and enforces authorization before retrieval and context construction. The historical security results must not be interpreted as tests of current cross-user isolation.
