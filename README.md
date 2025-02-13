### Autoencoders for complex- and real-valued autoencoder

***This repositorio contains the code related to paper "GNSS anomaly detection with complex-valued LSTM-based autoencoder":***

O. Savolainen, A. Elango, A. Morrison, N. Sokolova and L. Ruotsalainen, "GNSS Anomaly Detection with Complex-Valued LSTM Networks," 2024 International Conference on Localization and GNSS (ICL-GNSS), Antwerp, Belgium, 2024, pp. 1-7, doi: 10.1109/ICL-GNSS60721.2024.10578405.

## Paper Abstract
Global Navigation Satellite Systems (GNSS) serve many critical systems. Unfortunately, the GNSS based services are threatened by interference causing anomalies to the acquired signals. To protect the critical infrastructure, navigation signal quality should be monitored, anomalies immediately detected, isolated, and back-up solutions used. Previous GNSS anomaly detectors concentrate on one interference type only. Although methods based on deep learning are emerging, most work use convolutional neural networks, which are transcendent in processing spatially correlated data, such as images. However, GNSS data has temporal correlation, which requires suitable models such as Long Short-Term Memory (LSTM) networks. Traditionally, deep learning models have been trained using supervised methods requiring laborious labelling and therefore slowing down the modelling of complicated real-world phenom-ena. This paper presents, as far as we know, the first unsupervised LSTM based autoencoder for GNSS anomaly detection. LSTM autoencoders used in other domains process data in real or semi-complex domains and we claim that processing the signal at fully complex domain will improve the detection. Thereby, we present here the first fully complex-valued detector and test it with both real and complex-valued GNSS data. Our model in the real domain provides results that are comparable with the equivalent supervised method's 95% accuracy, outperforming 92% with our complex domain model. We claim that this lower performance is due to the implementation challenges which will be carefully discussed to accelerate the future research.

## Data
- Data is expected to be in a binary file (`.dat` or `.DAT`) including in-phase and quadrature (IQ) components. These two components should alternate as follows IQIQIQ...
    - In our experiments, the real-world data was collected as a part of the ARFIDAAS -project.
    - Generated data was collected with Orolia's signal simulator.
    - The data collected by us is not public available.
- For training the model, only clean data is used. That is, the detection is based on the reconstruction error in the prediction. Test data should contain both clean and samples with interference.

## Requirements
General machine learning and signal processing libraries:
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
There are two types of training possibilities: train either a real-valued or fully complex-valued network by choosing from the files ```run_complex.py``` or ```run_real.py```. The train principles are the same.
1. Add paths for you training and testing data for the functions named ```read_data_from_dat```.
2. Adjust hyperparameters in the beginning of the file to correspond your needs.
3. Run the code either from the console or if you have jupyter notebook installed and you Visual Studio Code, you can start training by clicking `Run below` above the first line which opens the interactive window.

## License
This code is licensed with a MIT licence.
