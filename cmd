python -m pip install --upgrade build
python -m build

python -m pip install --upgrade twine
twine upload dist/*