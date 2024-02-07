# ML_EEG_DEAP : Machine Learning for EEG Dataset : DEAP 
http://www.eecs.qmul.ac.uk/mmv/datasets/deap/

The full dataset description and download links are available in the website above. But in short ~ <br>
EEG was recorded from 32-channel scalp electrodes from subjects watching numerous 1-minute long videos and each participant rated the values in terms of a couple different factors - arousal, valence, like/dislike, dominance and familiarity.

<b> Goal : </b> Predict 'Emotions' from EEG

## Task Description

Full disclaimer, this was a freelance project that I took up when I was bored in my last semester of Bachelor's during the pandemic. (Yes, I'm uploading it 3 years later).
The task was pretty well defined - I had some clear directions on the execution, but code and method of implementation was entirely upto me (yay).

1. Decomposing signal into frequency bands using FIR filters
2. Feature extraction (choice of features detailed below)
3. Dimensionality reduction
   - Recursive Feature Elimination
   - PCA
   - LDA
4. Classification of emotion using (practically) every classifier (under the sun).
5. Compare accuracy and F1 across classifiers
6. Look at differences when using different splits of data
   - Female vs Male subjects
   - Random grouping (groups of 2,4,8)
  

# Execution

## Step 1: Process EEG data and save
Essentially, I decomposed the signal, extracted all the features and saved it as pickle files so I can create pipelines for the various models and execute them using a data generator instead of processing EEG data each time. (Big brain time)

- _1.1_Feature_Ext_Sample.ipynb_ : Shows the process and sample snippets of data when the decomposing is executed and features are extracted
- _1.2_FeatureExtraction.ipynb_ : Code for execution of process and save-to-pickle of data

<b> The Extracted Features are  - {SE1, SE2, O, HFD, Stat1,...., Stat8, PSD, LogPSD, Hemi, HdSE1, HdSE2, HdO}</b>

- SE1 - Shannon Entropy 1 - Traditional method
- SE2 - Shannon Entropy 2 - Spectral Entropy for Time series
- O - Oscillation Feature
- Stat1-8 : 8 Statistical Features
    - Mean
    - Standard Deviation
    - Skewness
    - Kurtosis
    - Mean of First Difference of Raw Signal
    - Mean of First Difference of Normalized Signal
    - Mean of Second Difference of Raw Signal
    - Mean of Second Difference of Normalized Signal
- PSD - Absolute power of the Power Spectral Density Spectrum
- LogPSD - Logarithm of Absolute power of the Power Spectral Density Spectrum
- Hemi - Hemispheric mean difference of raw signals 
- HdSE1, HdSE2, HdO - Hemispheric difference between the Shannon entropy and oscillation features
- HFD - Higuchi Fractal Dimension Feature

nFeatures = 18 for each frequency band <br>
nFeatures for each trial = 32 (electrode channels) x 5(Freq bands) x 18 features
                         <br> <b>=  2880 features per trial per patient


I also create a 'DataGenerator.py'. Essentially loads and combines data from different subjects as required by the 'splits' based on patient IDs or male/female tags etc. 

## Step 2: Make Pipelines, Run Combinations, Get Accuracies

Pipeline : 
![Alt text](/img/PipelineW.png)

Made a ton of Custom Pipelines, that was fun (sklearn is kinda amazing, but so am I). 

This is fairly straightforward and explained in the following three notebooks:
Each of the following notebooks looks at a different datasplit, but the execution and pipeline is the same in all three notebooks, its just separated so I didn't have to change a million things each time (even though I have coded it as such. This is also for easy reading/understanding)
If you just want an idea of how my pipeline works, look only at any one notebook. 

- _2.1_Gender_Dependent.ipynb_ : Male vs Female subjects
- _2.2_Subject_Dependent.ipynb_ : Scores for each Subject, Scores for random groups (sizes 2,4,8)
- _2.3_Subject_Independent.ipynb_ : Full Dataset together

Also, each notebook saves results into excel sheets as a table of accuracies and F1 scores, showing the scores for each  (Dimensionality reduction mode, ML model) pair.


# Extra Comments

- I also have a couple MATLAB codes that I used as help/inspiration (?) for the feature extraction functions.

 
