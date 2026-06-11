# LATTE configuration files

The JSON files in this directory are templates for LATTE experiments. They do
not include data, checkpoints, or institution-specific filesystem paths.

Use the files in `configs/examples/` as starting points for new data. They use
environment variables such as `${LATTE_WORK_ROOT}`, `${LATTE_DATA_ROOT}`, and
`${LATTE_REPO_ROOT}` for paths.

Example:

```bash
export LATTE_REPO_ROOT="$PWD"
export LATTE_DATA_ROOT="/path/to/your/data"
export LATTE_WORK_ROOT="/path/to/your/latte-runs"
```
