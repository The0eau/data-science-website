# Golden Template

[![CI Pipeline](https://github.com/The0eau/Golden-template/actions/workflows/main.yml/badge.svg)](https://github.com/The0eau/Golden-template/actions/workflows/main.yml)
[![CodeQL](https://github.com/The0eau/Golden-template/actions/workflows/codeql.yml/badge.svg)](https://github.com/The0eau/Golden-template/actions/workflows/codeql.yml)

This repository is a "Golden Template" for starting new projects. It comes pre-configured with a best-practice structure, security checks, and a CI/CD pipeline.

## ✨ Features

- **CI/CD Pipeline**: An automated workflow in `.github/workflows/main.yml` that handles:
  - **Linting**: Checks code formatting using [Prettier](https://prettier.io/).
  - **Testing**: Runs unit tests across multiple versions of Python and Node.js.
- **Automated Security Scans**:
  - **CodeQL**: Static analysis to find common vulnerabilities (`.github/workflows/codeql.yml`).
  - **Gitleaks**: Scans repository history for committed secrets.
- **Dependabot**: Automatically creates Pull Requests to keep your dependencies up to date (`.github/dependabot.yml`).
- **Standardized Templates**: Includes templates for bug reports, feature requests, and pull requests in the `.github` directory.
- **Clear Contribution Guidelines**: `CONTRIBUTING.md` provides a clear path for contributors.

## 🚀 How to Use This Template

1.  Click the **"Use this template"** button at the top of this page.
2.  Give your new repository a name and description.
3.  Choose whether to include all branches or just the `main` branch.
4.  Click **"Create repository from template"**.

You now have a new repository with all the files and configurations from this template!

## 🏁 Getting Started

1.  **Clone your new repository**:

    ```bash
    git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
    cd YOUR-REPOSITORY-NAME
    ```

2.  **Install dependencies**:
    This template is configured for both Python and Node.js.

    ```bash
    # For Python
    pip install -r requirements.txt

    # For Node.js
    npm install
    ```

    _Note: You may need to create `requirements.txt` or `package.json`._

3.  **Start coding!**
