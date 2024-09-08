# Test environments

This repository's tests and development automation tasks are organized using [tox], a command-line CI frontend for Python projects.  tox is typically used during local development and is also invoked from this repository's GitHub Actions [workflows](../.github/workflows/).

tox can be installed by running `pip install tox`.

tox is organized around various "environments," each of which is described below.  To run _all_ test environments, run `tox` without any arguments:

```sh
$ tox
```

Environments for this repository are configured in [`tox.ini`] as described below.

## Lint environment

The `lint` environment ensures that the code meets basic coding standards, including

- [_Black_] formatting style
- Style checking with [ruff], [autoflake], and [pydocstyle]
- [mypy] type annotation checker, as configured by [`.mypy.ini`]

To run:

```sh
$ tox -e lint
```

## Documentation environment

The `docs` environment builds the [Sphinx] documentation locally.

For the documentation build to succeed, [pandoc](https://pandoc.org/) must be installed.  Pandoc is not available via pip, so must be installed through some other means.  Linux users are encouraged to install it through their package manager (e.g., `sudo apt-get install -y pandoc`), while macOS users are encouraged to install it via [Homebrew](https://brew.sh/) (`brew install pandoc`).  Full instructions are available on [pandoc's installation page](https://pandoc.org/installing.html).

To run this environment:

```sh
$ tox -e docs
```

If the build succeeds, it can be viewed by navigating to `docs/_build/html/index.html` in a web browser.

[tox]: https://github.com/tox-dev/tox
[`tox.ini`]: ../tox.ini
[mypy]: https://mypy.readthedocs.io/en/stable/
[`.mypy.ini`]: ../.mypy.ini
[_Black_]: https://github.com/psf/black
[ruff]: https://github.com/charliermarsh/ruff
[autoflake]: https://github.com/PyCQA/autoflake
[pydocstyle]: https://www.pydocstyle.org/en/stable/
[Sphinx]: https://www.sphinx-doc.org/
