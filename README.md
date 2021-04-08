# GAN_Imputation_Galaxy_redshift
GAN missing value imputation with Astronomical data set

Authors: Kieran J. Luken , Ray P. Norris, X. Rosalind Wang , Laurence A. F. Park , Miroslav D. Filipovic 

Paper: Estimating Galaxy Redshift in Radio-Selected Datasets using Machine Learning

Link: https://github.com/kluken/Redshift-kNN-2021

This files contain different imputed files after splitting into training and testing datatset
Verify their eta(outlier) value to check which imputation is better and compare with other imputation technique.
We have used ATLAS redshift estimated dataset. We have referred Kieran's code for their modelling such as classification and regression.
We also refered Kieran's code for calculating outlier values and accuracy.


1. Mean Imputation 
2. Median Imputation
3. MICE Imputation
4. KNN Imputation
5. GAIN Imputation


For executing this file you should follow below steps.

1. Get the imputated file from each imputation type.
2. Place that file in appropiate path and do refer that path in the kNN_arg.py
3. DO check the method_no as per your imputation method.
4. Run the runALLTest.py file first that will internally called other files.
