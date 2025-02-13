### Autoencoders for complex- and real-valued autoencoder

***This repositorio contains the code related to paper "GNSS anomaly detection with complex-valued LSTM-based autoencoder":***

O. Savolainen, A. Elango, A. Morrison, N. Sokolova and L. Ruotsalainen, "GNSS Anomaly Detection with Complex-Valued LSTM Networks," 2024 International Conference on Localization and GNSS (ICL-GNSS), Antwerp, Belgium, 2024, pp. 1-7, doi: 10.1109/ICL-GNSS60721.2024.10578405.

## Abstract

## Data
- Data is expected to be in a binary file (`.dat` or `.DAT`) including in-phase and quadrature (IQ) components. These two components should alternate as follows IQIQIQ...
    - In our experiments, the real-world data was collected as a part of the ARFIDAAS -project.
    - Generated data was collected with Orolia's signal simulator.
    - The data collected by us is not public available.
- For training the model, only clean data is used. That is, the detection is based on the reconstruction error in the prediction.

## Requirements
General machine learning and signal processing librariers:
- PyTorch
- numpy
- pandas
- scipy
- scikit-learn

Plotting
- matplotlib
- seaborn

Note: in 2024, there was not yet support for all of the complex-valued operations in the CUDA environment, so cpu was used.
The code is tested to work with the latest versions of each library (2024).

## Training and evaluating the model
There are two types of 
1. Choose either ```run_complex.py``` or ```run_real.py```
2. Adjust hyperparameters and file names
3. Run the code

## License
This code is licensed with a MIT licence.
