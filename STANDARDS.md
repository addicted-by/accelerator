# PEP standards coverage:

| **PEP / Standard**                          | **Topic**                                                | **Rule system / Hook**                                                                 | **Example rules / checks**                                                                                               | **Reason for choice**                                                                                    |
| ------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **PEP 8** – Style Guide for Python Code     | Code style, indentation, line length, naming conventions | `ruff-check`, `black`                                                                  | - Line length <br>- Indentation 4 spaces<br>- Trailing whitespace removed<br>- Consistent line endings                   | Ensures uniform formatting and readability; automatically applied via Black and Ruff.                    |
| **PEP 257** – Docstring Conventions         | Docstrings                                               | `ruff-check`                                                                           | - Require triple quotes<br>- Docstring for functions/classes<br>- No missing docstrings                                  | Guarantees proper documentation coverage and standardizes docstring style.                               |
| **PEP 484** – Type Hints                    | Typing                                                   | `mypy`                                                                                 | - `--strict` mode enforces type annotations<br>- Detects type mismatches<br>- Ignores missing imports where not critical | Provides static type checking for safer, maintainable code; helps in larger projects like `accelerator`. |
| **PEP 518** – pyproject.toml                | Build System / Packaging                                 | `check-toml`, `prettier`                                                               | - Valid TOML syntax<br>- Consistent formatting                                                                           | Ensures the `pyproject.toml` is correct and readable, supporting Black, Ruff, and build tools.           |
| **PEP 263** – Source Code Encoding          | File encoding                                            | `ruff-check`                                                                           | - UTF-8 enforced (default)                                                                                               | Avoids encoding issues in source files, especially with docstrings/comments.                             |
| **PEP 561** – Type Information for Packages | Typing support in packages                               | `mypy` with `types-setuptools`                                                         | - Recognize type stubs in dependencies                                                                                   | Required for fully typed packages and correct external type resolution.                                  |
| **PEP 396** – Module Versioning             | Version declaration                                      | `check-added-large-files` / `check-yaml`                                               | - Validate `__version__` format                                                                                          | Helps maintain consistent versioning across submodules.                                                  |
| **PEP 420** – Namespace Packages            | Module structure                                         | `ruff-check`                                                                           | - Detect implicit namespace usage                                                                                        | Your `accelerator` submodules may be structured as namespace packages; Ruff ensures no illegal imports.  |
| **PEP 257 / 8 combination**                 | General hygiene                                          | `end-of-file-fixer`, `trailing-whitespace`, `check-case-conflict`, `mixed-line-ending` | - EOF newline<br>- No trailing whitespace<br>- No case conflicts<br>- Consistent line endings                            | Fixes common formatting issues that can break CI or cross-platform compatibility.                        |
| **Security / Secrets**                      | Secret management                                        | `detect-secrets`                                                                       | - Detect passwords, API keys, or tokens in code                                                                          | Prevents committing sensitive information to the repository.                                             |
| **Notebook / Jupyter hygiene**              | Notebooks                                                | `nbqa-black`, `nbqa-ruff`                                                              | - Autoformat `.ipynb` cells with Black<br>- Apply Ruff lint/fix in notebooks                                             | Ensures consistency of notebook code with main codebase standards.                                       |
| **Non-Python formatting standards**         | Markdown, YAML, JSON, Shell                              | `prettier`                                                                             | - Consistent indentation<br>- Proper line endings<br>- Quotation style in JSON/YAML                                      | Ensures all project files follow consistent formatting, reducing unnecessary diffs.                      |

# PEP standards diagram:

```text
                      ┌─────────────┐
                      │   PEP 8     │
                      └─────┬───────┘
                            │
      ┌─────────────────────┼──────────────────────┐
      │                     │                      │
  Formatting            Docstrings              Code hygiene
 Black, Ruff,         Ruff, nbQA           debug-statements,
 nbqa-black,           nbqa-ruff             mixed-line-ending
 end-of-file-fixer,
 trailing-whitespace

                      ┌─────────────┐
                      │   PEP 257   │
                      └─────┬───────┘
                            │
                     Docstring enforcement
                      (Ruff, nbqa-ruff)

                      ┌─────────────┐
                      │  PEP 484    │
                      └─────┬───────┘
                            │
                     Type checking
                        (Mypy)

                      ┌─────────────┐
                      │  PEP 561    │
                      └─────┬───────┘
                            │
                Type stubs recognition
                        (Mypy)

                      ┌─────────────┐
                      │  PEP 420    │
                      └─────┬───────┘
                            │
                Namespace package detection
                        (Ruff)

                      ┌─────────────┐
                      │  PEP 518    │
                      └─────┬───────┘
                            │
                Build/config validation
                 (check-toml, check-yaml)

                      ┌─────────────┐
                      │ Security    │
                      └─────┬───────┘
                            │
         ┌──────────────────┴─────────────────┐
         │                                    │
   Secrets detection                       Security checks
   (detect-secrets)                        (Bandit)

                      ┌─────────────┐
                      │  Non-Python │
                      └─────┬───────┘
                            │
               Formatting (Markdown, YAML, JSON, shell)
                           (Prettier)
```
