# Geometric deep learning implementation

Code to build and test neural networks bases on geometric deep learning

## Framework installation

To install the framework you need anaconda and git on a linux machine. In a terminal type:
1. Clone the repository:
  ```
  git clone git@github.com:andrex-naranjas/geometric-nn.git
  ```
2. Access the code:
  ```
  cd geometric-nn
  ```
3. Install the conda enviroment:
  ```
  conda env create -f config.yml
  conda activate geometric
  conda develop .
  ```
3.1 Update the conda enviroment:
   ```
   conda env update --file config.yml --prune
   ```

## Run jupyter notebooks
To run the notebooks on a linux machine, in a terminal type:
1. Open jupyter notebook and select the port you want to connect:
   ```
   jupyter notebook --no-browser --port=8000
   ```

2. If you are working on a remote machine, create a ssh tunnel:
   ```
   ssh -L 8000:localhost:8000 yourname@remote.host.address
   ```

3. Copy the the url in you favorite web browser
