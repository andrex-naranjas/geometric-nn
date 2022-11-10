# Geometric deep learning implementation

Code to compute bottom baryon spectra and decay widths. A fit is performed to obtain the model parameters. Errors are propagate via bootstrap Monte Carlo Gaussian sampling.

## Framework installation

#No no eres
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