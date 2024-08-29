pip uninstall DimRed
rm -rf build
rm -rf DimRed.egg-info
pip install .
cd docs
make html
cd ..
