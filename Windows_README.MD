(Note that you need to do this well before coming to the workshop when you have data connection)

Install with Windows:
------------------------------

Python compatible with Tensorflow (3.5 or 3.6) installations can be found 
	https://www.python.org/downloads/
	Make sure to select "Add Python 3.5 to PATH" (for 3.5 installation)
	"Add Python 3.5 to PATH" (this is for 3.6 installation).
	
Please see that you install the 64-bit version of Python.
	
Install git scm and bash shell.
         https://gitforwindows.org/

***Now open the git bash terminal***
--------------------------------------------------

1. Use command
	$ pip3 install --upgrade tensorflow
this will install tensorflow
(For now install only cpu version. If you would like GPU version, better to use nv-docker images provided by Nvidia)

2. Package can be downloaded from 
	https://github.com/ieee-mlw/ieee_ml.git

Or use command 
	$ git clone https://github.com/ieee-mlw/ieee_ml.git

3. Once you checked out/download the above software package, Use commands

	$ cd ieee_ml
	
	This directory has two model directories (problem_sets and guided_examples) along with several text files.
	requirements.txt specifies required libraries for executing the models in the model diretories.
	
	$ pip install -r requirements.txt
	
(requirements.txt is in the directory of ieee_ml you just checked out)

4. Before initiating the jupyter notebook command; run the start.sh bash file from the home directory.

	$ chmod +x start.sh
	
	$ ./start.sh

This starts the Jupyter notebook session on a localhost port while simultaneously indicating the URL to access Tensorboard for debugging the current script and also visualize the learning process at different stages during training.


	
5. Running Python files	[Optional]

	This package contains IPython notebooks (jupyter) and their executable python (.py) files. The .py files are obtained using the command "jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb"
	So if you would like to work with .py files after updating .ipynb file, you need to run the above command to get corresponding .ipynb file. As an example,
	
	$ cd guided_examples
	
        $ jupyter nbconvert --to python kmeans.ipynb
	
	$ python kmeans.py
	
	If you would like to run with debug mode enabled
	$ python kmeans.py --debug
	
	
Potential Installation Issues
----------------------------------------

Tensorflow install issues for Windows
   
We need a MSVCP140.DLL file to use tensorflow through python in Windows.
This file can be installed during the installation of Visual C++ 2015
redistributable (link: https://www.microsoft.com/en-us/download/details.aspx?id=53587 ) 


