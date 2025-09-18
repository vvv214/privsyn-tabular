# PrivSyn Tabular Documentation

Welcome to the MkDocs-powered handbook for the PrivSyn tabular synthesis project. Use the navigation on the left to jump straight into frontend internals, backend APIs, testing strategy, deployment notes, and privacy considerations.

## Quick Links

- [Repository README](../README.md) – high-level overview, quick-start commands, and architecture diagrams.
- [Frontend Guide](frontend.md) – React/Vite structure, metadata editors, and component testing.
- [Backend Guide](backend.md) – FastAPI endpoints, synthesis orchestration, and data flow.
- [Testing Playbook](testing.md) – pytest conventions, Playwright E2E flows, and focused command examples.
- [Deployment Checklist](deployment.md) – Docker build, Cloud Run notes, and prod env variables.
- [Privacy Notes](privacy.md) – epsilon/delta guidance and rho-CDP budgeting for PrivSyn vs. AIM.

## Getting Started Locally

```bash
pip install mkdocs
mkdocs serve
```

Browse the site at <http://127.0.0.1:8000/> to view these pages with navigation search and table-of-contents support.

## Contributing

Keep project documentation under the `docs/` directory. When you add a new Markdown file, update `mkdocs.yml` so it appears in the navigation, and run `mkdocs serve` to verify formatting locally.
