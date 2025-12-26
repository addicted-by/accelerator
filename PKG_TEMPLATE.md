# Package Name

Short tagline or one-liner describing the package.

---

## Table of Contents

- [Overview](#overview)
- [Canonical Structure](#canonical-structure)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup](#setup)
- [Development](#development)
- [CI/CD](#cicd)
- [Ownership](#ownership)
- [License](#license)
- [C&A Team](#ca-team)

---

## Overview 📝

**What:** Describe what the project does.
**Why:** Explain the problem it solves or motivation.
**When / Where:** When and where to use it.
**For Whom:** Target user.

---

## Canonical Structure 📂

```bash
.
├── .envs/                # Environment configs
├── src/                  # Project / package source code (importable)
├── tests/                # Unit & integration tests
├── docs/                 # Documentation
│   └── adr/              # Architecture Decision Records (optional)
├── scripts/              # Dev, CI, maintenance scripts
├── .env.example          # Non-sensitive environment variables
└── README.md             # This file
```

## Architecture 🏗️

High-level design description. Add diagrams if possible.

## Requirements ⚙️

**Python Version**:

**Dependencies**: `requirements.txt`

**Environment**:

## Setup 🛠️

1. Clone repository

```bash
git clone git@repo-url:project.git
cd project
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Set up environmental variables (if necessary)

```bash
cp .env.example .env
```

4. Project related setup

## Development 💻

How to test, lint, format, debug

## CI/CD

Which pipelines run and why

## Ownership 👥

Project code owners
