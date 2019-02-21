.PHONY: conda_dev0 
conda_dev0:
	# if you don't want to contaminate your python libraries
	#     make sure to create an python3 environement before you run this
	conda install numpy scipy pandas pip git cython jupyter seaborn scikit-learn --yes
	conda install -c conda-forge ipyvolume --yes
	# pip install git+https://github.com/shifwang/staNMF.git 
	# pip install -q ray psutil bokeh

