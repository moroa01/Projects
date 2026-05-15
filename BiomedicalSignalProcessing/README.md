# Biomedical Signal processing project: ECG signal decomposition using Fourier analysis
### This is a project for the [Biomedical Signal processing](https://www.unimi.it/it/corsi/insegnamenti-dei-corsi-di-laurea/2025/biomedical-signal-processing) course of master's degree in Computer Science at the Università degli Studi di Milano

### Project Overview

* **Project Description**: Implementation of an ECG signal decomposition tool using **Discrete Fourier Series**. The project aims to isolate the T-wave from the QRS-complex by reconstructing signals with specific cutoff frequencies. It includes a comparative analysis of $\ell_p$ minimization techniques ($\ell_1$ vs. $\ell_2$) to handle noise (such as spike noise) and improve diagnostic accuracy in cardiac monitoring.

* **Code**: the simulations and practical examples are in [**this**](https://github.com/moroa01/Projects/blob/main/BiomedicalSignalProcessing/Simulations.m) file <br>
* **Presentation**: the presentation (PowerPoint) on the project is in [**this**](https://github.com/moroa01/Projects/blob/main/BiomedicalSignalProcessing/Presentation.pptx) file 

### The signals used for testing the algorithm are:
* #### A standard ECG provided by the professor in the course
* #### An ECG of a patient with Ventricular hypertrophy from the PTB-XL dataset [here](https://physionet.org/content/ptb-xl/1.0.1/)
* #### An ECG of a patient with Atrial fibrillation from the MIT-BIH dataset [here](https://physionet.org/content/afdb/1.0.0/)
* #### Some step signals to test some properties
