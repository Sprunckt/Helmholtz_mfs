# Helmholtz_mfs

Code for the paper "From sound to shape: polygonal room reconstruction via shape optimization".

Experiments performed with python 3.10.8, numpy 2.2.5, scipy 1.15.2.

## Usage

*simulation.py* contains the Simulation class that solves the Helmholtz equation using the Method of Fundamental Solutions (run ```python3 simulation.py``` for an example.)

*shape_opt.py* handles the shape gradient descent algorithm. The script *shape_descent_example.py* provides a running example that : 
* Creates a target polygon
* Simulates observations for this polygon
* Sets a random initialization 
* Launches a gradient descent for this problem

Note that the number of initializations, observed frequencies and gradient steps are set purposely low to allow the script to run through (should be a few minutes on a laptop). A convergence to the ground truth shape is not expected here, but increasing the number of random initializations/frequencies/gradient steps decreases the probability of converging to a local minimum. Note also that the cost function might increase at certain iterations if the line search fails due to a large step size, or if the maximum observed frequency is increased. Increasing the number of random initialization and the initial gradient steps for each initialization also decreases the chances of the optimization failing later on.