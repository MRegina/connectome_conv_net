# connectome_conv_net
The repository contains conv_net.py, the python (TensorFlow) source code of the convolutional neural network for connectome based classification. It also contains simulate_data.py, that reads in the uploaded .npy files with base and artificially modified connectomes, and creates a simulated dataset with tensors ready for the convolutional network. The simulation is the following: we took a bese connectome of a healthy adult, and replaced 10 rows and columns from the connectome of another subject (ROI IDs in ROI_IDs.npy). Individual instances were created by adding Gaussian-noise with a weight of 5 to these two connectomes (deatiled description in the research paper). The tensors are 150x499x499x1 numpy arrays, the first dimension encodes instances (subjects), the second and third dimensions, i.e. the 499x499 matrices are the individual connectomes. The fourth dimension encodes "channels" (like RGB in case of images). 

conv_net.py can read in tensors, and contains the necessary functions to create cross-validation folds, randomize and normalize tensors etc. as well as the convolutional network model in tensorflow. The code outputs a .npz file that contains a vector of predicted labels and a vector of true labels (in the order described in the IDs variable)
 
  
RESEARCH PAPERS USING THIS SOFTWARE:
- preprint:
Meszlényi, R., Buza, K., and Vidnyánszky, Z. (2017). Resting state fMRI functional connectivity-based classification using a convolutional neural network architecture. ArXiv170706682 Cs Stat. Available at: http://arxiv.org/abs/1707.06682.
- submitted to Frontiers in Neuroinformatics

