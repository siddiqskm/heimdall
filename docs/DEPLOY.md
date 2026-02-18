# Publishing heimdall as a Python package

## 1. GitHub

- Repo: [github.com/siddiqskm/heimdall](https://github.com/siddiqskm/heimdall)
- Push the project:

  ```bash
  git remote add origin https://github.com/siddiqskm/heimdall.git
  git push -u origin main
  ```

## 2. (Optional) Publish to PyPI

So that `pip install heimdall` works from PyPI:

```bash
poetry build
poetry publish
```

Consumers can then `pip install heimdall` and use the package: instantiate `Classifier`, `LabelDwell`, `Embedder`, and call `decide`, `route` as needed.
