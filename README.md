Catalog Statistics and Comparison
=================================

Generate plots and statistics for single catalogs and for catalog-catalog comparisons. `QCreport.py` generates a catalog summary and figures concerning various data within the catalog. It is run by typing `QCreport.py <catalog> <startyear> <endyear>`; e.g. `QCreport.py us 2010 2012`. The resulting figures and data can be found in the generated folder with the format `<catalog><startyear>-<endyear>`; e.g. `us2010-2012`.

Tested and working in Python 2.7 and 3.6.

Required Python packages
------------------------
1. [NumPy](http://www.numpy.org)
2. [Matplotlib](https://matplotlib.org)
3. [Pandas](http://pandas.pydata.org)
4. [Basemap](https://matplotlib.org/basemap)
