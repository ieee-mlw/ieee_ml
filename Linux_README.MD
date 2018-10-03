(Note that you need to do this well before coming to the workshop when you have data connection)
Tensor flow installation
---------------------------------
https://www.tensorflow.org/install/install_linux#InstallingVirtualenv

 $ sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
  
  
 $ pip install --upgrade pip
  
 $ mkdir ~/tensorflow  # somewhere to work out of
    
 $ cd ~/tensorflow
    
 $ virtualenv --system-site-packages -p python3 venv # Use Python 3.n
    
 $ source ~/tensorflow/venv/bin/activate      # bash, sh, ksh, or zsh
  
When the Virtualenv is activated, the shell prompt displays as (venv) $.
Within the active virtual environment, upgrade pip:

(venv)$ pip install --upgrade pip

You can install other Python packages within the virtual environment without affecting packages outside the virtualenv.

(venv)$ pip install --upgrade tensorflow

Use pip list to show the packages installed in the virtual environment. Validate the install and test the version:

(venv)$ python -c "import tensorflow as tf; print(tf.__version__)"

Success: TensorFlow is now installed.

While here do additional installation steps and downloading repos as needed.

Additional Steps while at (venv) $.
-------------------------------------------------
3. Package can be downloaded from 
	https://github.com/ieee-mlw/ieee_ml.git

Or use command 
	(venv) $ git clone https://github.com/ieee-mlw/ieee_ml.git

4. Use command 

        (venv) $ cd ieee_ml
This directory has two model directories (problem_sets and guided_examples) along with several text files. requirements.txt specifies required libraries for executing the models in the model diretories.
	
        (venv) $ pip install -r requirements.txt
	
(requirements.txt is in the directory of ieee_ml you just checked out)

5. Before initiating the jupyter notebook command; run the start.sh bash file

	(venv) $ bash start.sh
	
This starts the Jupyter notebook session on a localhost port while simultaneously indicating the URL to access Tensorboard for debugging the current script and also visualize the learning process at different stages during training.
	
6. Running Python files	[Optional]

This package contains IPython notebooks (jupyter) and their executable python (.py) files. The .py files are obtained using the command "jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb" So if you would like to work with .py files after updating .ipynb file, you need to run the above command to get corresponding .ipynb file. As an example,

 	(venv) $ cd guided_examples

 	(venv) $  jupyter nbconvert --to python kmeans.ipynb
 
 	(venv) $  python kmeans.py

If you would like to run with debug mode enabled 

 	(venv) $  python kmeans.py --debug



After your work is done, Use the deactivate command to stop the Python virtual environment.




