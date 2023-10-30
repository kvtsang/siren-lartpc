# Documentation

[![Documentation Status](https://readthedocs.org/projects/siren-lartpc/badge/?version=latest)](https://siren-lartpc.readthedocs.io/en/latest/?badge=latest)


We use Sphinx to generate the documentation, and Readthedocs.io to host it at https://siren-lartpc.readthedocs.io/en/latest/.
In theory the online documentation gets built and updated automatically every time the source branch changes.

## Writing docstrings
If possible, let us try to consistently use NumPy style. See [Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html) and [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) style guides.

### Documenting a generic function
```
def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    """
    return True
```

## Building the documentation

First, install necessary pieces
```
$ cd docs
$ pip install -r requirements.txt
```

If you make a change in the existing file, simply try:
```
$ cd docs
$ make html
```

If you introduced a new file or changed the directory structures, try:
```
$ cd docs
$ sphinx-apidoc -f -M -e -T --templatedir templates/apidoc/ -o ./source ../slar
$ make html
```

Note: `sphinx-apidoc` generates automatically a .rst file for each Python file
it detected (recursively). It needs a `__init__.py` file in a folder for
it to be recognized as a Python package.

Then open the file `docs/_build/html/index.html` in your favorite browser.

### On ReadTheDocs.org
The configuration for this build is in `../.readthedocs.yaml`.
The dependencies used by the build are in `requirements_rtd.txt`.