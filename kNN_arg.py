import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime
import functions
from sklearn.model_selection import KFold 
from tqdm import tqdm
import scipy.stats as stats
import os
import pylab
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
import pickle
from sklearn.metrics import adjusted_mutual_info_score
import sys, argparse
from sklearn.linear_model import LinearRegression,LogisticRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import metric_learn
from sklearn import metrics
import pandas as pd
import shutil


plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.axisbelow'] = True

def kNN_classification(kVal, xValsTrain, xValsTest, yValsTrain, yValsTest, distType):
    if distType < 5:
        neigh = KNeighborsClassifier(n_neighbors = kVal, p = distType)
    elif distType == 99:
        neigh = KNeighborsClassifier(n_neighbors = kVal, metric = "mahalanobis", metric_params={"V":np.cov(xValsTrain, rowvar=False)})
    neigh.fit(xValsTrain,np.squeeze(yValsTrain.astype(str)))
    predictions = neigh.predict(xValsTest.astype(str))
    return predictions.astype(np.float32).ravel(), neigh.score(xValsTest,np.squeeze(yValsTest.astype(str))).astype(np.float32)


def kNN(kVal, xValsTrain, xValsTest, yValsTrain, yValsTest, distType):
    if distType < 5:
        neigh = KNeighborsRegressor(n_neighbors=kVal, p=distType)
    elif distType == 99:   
        neigh = KNeighborsRegressor(n_neighbors=kVal, metric = "mahalanobis", metric_params={"V":np.cov(xValsTrain, rowvar=False)})
    neigh.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = neigh.predict(xValsTest)
    return predictions.ravel(), neigh.score(xValsTest,np.squeeze(yValsTest))

def linRegress(xValsTrain, xValsTest, yValsTrain, yValsTest):
    linModel = LinearRegression()
    linModel.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = linModel.predict(xValsTest)
    return predictions.ravel(), linModel.score(xValsTest,np.squeeze(yValsTest))

def logRegress(xValsTrain, xValsTest, yValsTrain, yValsTest):
    logModel = LogisticRegression()
    logModel.fit(xValsTrain,np.squeeze(yValsTrain).astype(str))
    predictions = logModel.predict(xValsTest)
    return predictions.astype(np.float32).ravel(), logModel.score(xValsTest,np.squeeze(yValsTest).astype(str)).astype(np.float32)

def randomForestClass(treeVal, xValsTrain, xValsTest, yValsTrain, yValsTest, randomState = False):
    if type(randomState) == bool:
        randomState = 42
    neigh = RandomForestClassifier(treeVal, random_state=randomState)
    neigh.fit(xValsTrain,np.squeeze(yValsTrain.astype(str)))
    predictions = neigh.predict(xValsTest.astype(str))
    return predictions.astype(np.float32).ravel(), neigh.score(xValsTest,np.squeeze(yValsTest.astype(str))).astype(np.float32)


def randomForestRegress(treeVal, xValsTrain, xValsTest, yValsTrain, yValsTest, randomState=False):
    if type(randomState) == bool:
        randomState = 42
    neigh = RandomForestRegressor(treeVal, random_state=randomState)
    neigh.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = neigh.predict(xValsTest)
    return predictions.ravel(), neigh.score(xValsTest,np.squeeze(yValsTest))

def lassoRegress(xValsTrain, xValsTest, yValsTrain, yValsTest):
    neigh = LassoCV()
    neigh.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = neigh.predict(xValsTest)
    return predictions.squeeze(), neigh.score(xValsTest,np.squeeze(yValsTest))

def ridgeRegress(xValsTrain, xValsTest, yValsTrain, yValsTest):
    neigh = RidgeCV(np.arange(0.1,10,0.1))
    neigh.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = neigh.predict(xValsTest)
    return predictions.ravel(), neigh.score(xValsTest,np.squeeze(yValsTest))



def metricLearnRegression(xVals, yVals):
    model = metric_learn.MLKR()
    #model = metric_learn.MLKR(num_dims=xVals.shape[1])
    model.fit(xVals, yVals.ravel())
    return model


def binDataFunc(redshiftVectorTemplate, numBins, maxRedshift = 1.5, binEdges = False, newZ = False):
    redshiftVector = redshiftVectorTemplate
    sortedRedshift = np.sort(redshiftVector)
    
    if type(binEdges) == bool:
        numPerBin = sortedRedshift.shape[0]//numBins #Integer division!
        binEdges = [0]
        for i in range(1, numBins):
            binEdges.append(i * numPerBin)
        binEdges.append(sortedRedshift.shape[0]-1)
        binEdges = sortedRedshift[binEdges]
        binEdges[-1] = maxRedshift
    

    if type(newZ) == bool:
        newZ = []
        for i in range(1, numBins + 1):
            if i < numBins:
                newZ.append(np.median([binEdges[i-1], binEdges[i]]))
            else:
                newZ.append(np.median(redshiftVector[np.where((redshiftVector >= binEdges[i-1]) & (redshiftVector < np.max(sortedRedshift)))[0]])) 
    for i in range(1, numBins + 1):
        if i < numBins:
            if i == 1:
                redshiftVector[np.where((redshiftVector < binEdges[i]))[0]] = newZ[i-1]
            else:
                redshiftVector[np.where((redshiftVector >= binEdges[i-1]) & (redshiftVector < binEdges[i]))[0]] = newZ[i-1]
        else: 
            redshiftVector[np.where((redshiftVector >= binEdges[i-1]))[0]] = newZ[i-1]
    return redshiftVector, binEdges, newZ

def plotNormConfusionMatrix(confusion, newZ, binEdges):
    # Rotating the confusion matrix, and creating the matrix to be normalised
    confusion_norm = np.rot90(confusion.astype(float), 1)
    # Plotting the normalised confusion matrix
    for i in range(confusion_norm.shape[0]):
        if (np.sum(confusion_norm[:,i]) != 0):
            confusion_norm[:,i] = confusion_norm[:,i]/np.sum(confusion_norm[:,i])
    fig, ax = plt.subplots()
    plt.xlabel(r"Spec$_z$")
    plt.ylabel(r"Photo$_z$")
    # Plot the "Image"
    im=ax.imshow(confusion_norm)
    # Set the ticks at the right values
    print("xtrick", confusion_norm.shape[0])
    print("ytrick", confusion_norm.shape[0])
    ax.set_xticks(np.arange(confusion_norm.shape[0]))
    ax.set_yticks(np.arange(confusion_norm.shape[0]))
    # ... and label them with the respective list entries
    labels = np.round(newZ,decimals=2).astype(str)
    print("xlabels", labels)
    labels[0] = "<" + np.round(binEdges[1],decimals=2).astype(str)
    labels[-1] = ">" + np.round(binEdges[-2],decimals=2).astype(str)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(np.flip(labels,axis=0))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    for i in range(confusion_norm.shape[0]):
        for j in range(confusion_norm.shape[0]):
            text = ax.text(j, i, np.round(confusion_norm[i, j],decimals=2), ha="center", va="center", color="w")
    plt.tight_layout()
    plt.savefig("confusionMatrix.pdf")

def plotScaledConfusionMatrix(realYVals, predYVals, binEdges, newZ):
    plt.figure(123)
    H,yedges, xedges = np.histogram2d(np.squeeze(predYVals), np.squeeze(realYVals), bins=(binEdges,binEdges))
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    X,Y = np.meshgrid(xedges, yedges)
    for i in range(H.shape[1]):
        H[:,i] = H[:,i] / np.sum(H[:,i])
    plt.pcolor(X, Y, H)
    plt.xlabel(r"Spec$_z$")
    plt.ylabel(r"Photo$_z$")
    labels = np.round(newZ,decimals=2).astype(str)
    labelsLocation = np.array(newZ).astype(float)
    labelsLocation[-1] = np.median([binEdges[-1], binEdges[-2]])
    labels[0] = "< " + np.round(binEdges[1],decimals=2).astype(str)
    labels[-1] = "> " + np.round(binEdges[-2],decimals=2).astype(str)
    loc,label = plt.xticks()
    plt.xticks(labelsLocation, labels,rotation=90)
    plt.yticks(labelsLocation, labels)
    plt.colorbar()
    plt.grid()
    plt.tight_layout()
    plt.savefig("scaledConfusion.pdf")


def mad( data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)

def plotData( specZ, predZ, plt, stats, pylab, error = False):
    num = specZ.shape[0]
    residual=(specZ-predZ)/(1+specZ)
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 14})
    
    fig, [ax,ay] = plt.subplots(2, sharex=True,  gridspec_kw = {'height_ratios':[2, 1]})
    fig.set_figheight(9)
    fig.set_figwidth(6)
    sizeElem=2
    if type(error) == bool:
        cax=ax.scatter(specZ, predZ, edgecolor='', s=sizeElem, cmap="gray", color="black", label='Not AGN')
        cay=ay.scatter(specZ, residual, edgecolor='', s=sizeElem, cmap="gray", color="black", label='Not AGN')
    else:
        cax=ax.errorbar(specZ, predZ, yerr = error, color="black", ms = sizeElem, lw = 1, fmt=".", label='Not AGN', alpha=0.2)
        cax=ax.scatter(specZ, predZ, edgecolor='', s=sizeElem, cmap="gray", color="black", label='Not AGN')
        cay=ay.scatter(specZ, residual, edgecolor='', s=sizeElem, cmap="gray", color="black", label='Not AGN')

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #    ncol=2, fancybox=True, shadow=True)
    
    ax.plot([0,4],[0,4], 'r--',linewidth=1.5)
    ax.plot([0,4],[0.15,4.75], 'b--',linewidth=1.5)
    ax.plot([0,4],[-.15,3.25], 'b--',linewidth=1.5)
    ay.plot([0,4],[0,0], 'r--',linewidth=1.5)
    ay.plot([0,4],[0.15,.15], 'b--',linewidth=1.5)
    ay.plot([0,4],[-.15,-.15], 'b--',linewidth=1.5)
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    ax.axis([0,4,0, 4.6])
    ay.axis([0,4,-.5, .5])

    outNum=100*len(residual[np.where(abs(residual)>0.15)])/len(residual)
    sigma=np.std(residual)
    nmad=1.4826*mad(residual)
    skew=stats.skew(residual)
    kurtosis=stats.kurtosis(residual)

    xlab=.3
    ylab=3.7
    step=-.3
    ax.text(xlab, ylab, r'$N='+str(num)+'$')
    ax.text(xlab, ylab+ step, r'$\sigma='+str(round(sigma, 2))+r'$')
    ax.text(xlab, ylab+ 2*step,        r'$NMAD='+str(round(nmad, 2))+r'$')
    ax.text(xlab, ylab+ 3*step,        r'$\eta='+str(round(outNum, 2))+r'\%$', fontsize=14)
    #ax.text(xlab, ylab+ 4*step,        r'$kurtosis='+str(round(kurtosis[0], 2))+r'$')
    #ax.text(xlab, ylab+ 5*step,        r'$skew='+str(round(skew[0], 2))+r'$')
    ax.set_ylabel('$z_{photo}$')
    ay.set_ylabel(r'$\frac{z_{spec}-z_{photo}}{z_{spec}+1}$')
    ax.grid()
    ay.grid()
    pylab.xlabel('$z_{spec}$')
    F = pylab.gcf()
    plt.tight_layout()
    F.savefig("resultsPlot.pdf", dpi = (200))


def main():
    hdul = fits.open("deep.fits")
    hdulData = hdul[1].data

    print(hdulData)
    parser = argparse.ArgumentParser(description="This script runs kNN with the required parameters. You should look at those.")
    parser.add_argument("-c", "--catType", nargs=1, required=True, type=int, help="0 for DEEP, 1 for WIDE - Catalogue Type, 2 for complete dataset, including missing values")
    parser.add_argument("-f", "--fillMethod", nargs=1, required=False, type=int, help="0 for replacement of missing values with the column mean. 1 for column median.") 
    parser.add_argument("-t", "--testType", nargs=1, required=True, type=int, help="0 for normal, 1 for sub-field test with train = ELAIS-S1, 2 = sub-field test with train = eCDFS") 
    parser.add_argument("-C", "--colsType", nargs=1, required=True, type=int, help="#0 for radio, sIndex, 3.6, 4.5, 5.8, 8.0, g, r, i, z. 1 for radio, 3.6, 4.5, 5.8, 8.0, g, r, i, z. 2 for radio, 3.6, 4.5, g, r, i, z") 
    parser.add_argument("-d", "--distType", nargs=1, required=True, type=int, help="1 for Manhattan, 2 for Euclidean, 99 for Mahalanobis") 
    parser.add_argument("-b", "--bootstrapSize", nargs=1, required=False, type=int, help="Number of bootstrap intervals. Do not use if you don't want bootstrap") 
    parser.add_argument("-p", "--preBin", nargs=1, required=False, type=bool, help="Should the data be pre-binned? Don't enter for no") 
    parser.add_argument("-P", "--postBin", nargs=1, required=False, type=bool, help="Should the data be post-binned? Don't enter for no") 
    parser.add_argument("-z", "--classification", nargs=1, required=False, type=bool, help="Classification or regression. True for classification Don't enter for no") 
    parser.add_argument("-m", "--metricLearn", nargs=1, required=False, type=bool, help="Should metric learning be used? Don't enter for no")
    parser.add_argument("-M", "--method", nargs=1, required=False, type=int, help="The type of ML to use. Nothing for kNN, 1 Linear Regression, 2 for Random Forest, 3 for lasso regression, 4 for ridge regression") 


    args = vars(parser.parse_args())

    

    #Data, Tests and Columns to use
    catType = args["catType"][0]
    testType = args["testType"][0]
    colsType = args["colsType"][0]
    distType = args["distType"][0]
    FailureLimit = 0.15
    nSplits = 10 #Used in k-Fold Cross Validation. 

    if args["fillMethod"] == None:
        fillMethod = 0
    else:
        fillMethod = args["fillMethod"][0] 
    if args["bootstrapSize"] == None:
        bootstrapSize = False
    else:
        bootstrapSize = args["bootstrapSize"][0] 
    binData = 15
    if args["postBin"] == None:
        postBin = False
    else:
        postBin = args["postBin"][0] 
    if args["preBin"] == None:
        preBin = False
    else:
        preBin = args["preBin"][0]
    if args["classification"] == None:
        classification = False
    else:
        classification = args["classification"][0]
    if args["metricLearn"] == None:
        metricLearn = False
    else:
        metricLearn = args["metricLearn"][0] 
    if args["method"] == None:
        MLMethod = 0
    else:
        MLMethod = args["method"][0] 
    if MLMethod == 2:
        neighboursList = range(2,60) #Using Random Forest. Should be different!
    elif MLMethod == 0:
        neighboursList = range(2,20) 
    else:
        neighboursList = [0] 

    folderpath = "cat-" + str(catType).strip() + "_fillMethod-" + str(fillMethod).strip() + "_testType-" + str(testType).strip() + "_colsType-" + str(colsType).strip() + "_distType-" + str(distType).strip()
    folderpath = folderpath + "_boot-" + str(bootstrapSize).strip() + "_preBin-" + str(preBin).strip() + "_postBin-" + str(postBin).strip() 
    folderpath = folderpath + "_class-" + str(classification).strip() + "_metricLearn-" + str(metricLearn).strip() + "_MLMethod-" + str(MLMethod)

    if not os.path.exists("Results"):
        os.makedirs("Results")
    os.chdir("Results")
    startTime = datetime.now()
    # if not os.path.exists(now.strftime("%d-%m-%Y")):
    #     os.makedirs(now.strftime("%d-%m-%Y"))
    # os.chdir(now.strftime("%d-%m-%Y"))
    if not os.path.exists(folderpath):
        os.makedirs(folderpath) 



    #Individual switches to modify "small" details
    logZ = False #Set to True if z should be log(z)
    #NOTE!!! There is no check on the below redshift modifications to make sure selections make sense!
    maxRedshift = None #Flag to set max redshift to keep
    minRedshift = None #Flag to set min redshift to keep
    np.random.seed(42) 

    useLogRadio = False #Set to true to take the log of radio data. False otherwise
    useColoursOptical = False #Set to True to use Optical Colours instead of Magnitudes
    useIRMagnitudes = False #Set to True to use log(IRFlux)
    useColoursIR = False #Set to True to use IR Colours instead of Fluxes (Implies True to above)
    standardiseXVals = True #Set to True to standardise the X-Values (x_i - x_mean) / x_sd


        
        
    

    if catType == 0:
        # catalogue = "../ATLAS_CATALOGUE/ATLAS-SWIRE-GRC-merged-2017-11-07.fits"
        catalogue = "deep.fits"
        # catalogue = "/bigdata/users/postgrad/kluken/Masters/ATLAS_CATALOGUE/ATLAS_Reduced_NoSindex.fits"
    elif catType == 1:
        catalogue = "wide.fits"
        #catalogue = "/bigdata/users/postgrad/kluken/Masters/ATLAS_CATALOGUE/ATLAS_EMU_3.6_4.5_DES.fits"
    elif catType == 2:
        catalogue = "missing.fits"



    if colsType == 0:
        dataCols = ["z","Sp2","Sindex","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"]
        dataType = [0,1,2,3,3,3,3,4,4,4,4]
    elif colsType == 1:
        dataCols = ["z","Sp2","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"]
        dataType = [0,1,3,3,3,3,4,4,4,4]
    elif colsType == 2:
        dataCols = ["z","Sp2","flux_ap2_36","flux_ap2_45","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"]
        dataType = [0,1,3,3,4,4,4,4]



    #Create instance of preprocessing class to clean data
    preprocess = functions.DataProcessing()


    #Open Fits Catalogue
    print(catalogue)
    print(folderpath)


    # hdul = fits.open("missing.fits")
    # hdulData = hdul[1].data
    # print("Time taken to open fitsFile: " + str(datetime.now() - startTime))
    os.chdir(folderpath)


    if os.path.isfile("resultsPlot.pdf"):
        print(folderpath + " already complete")
        sys.exit()


    #Create catalogueData array from the redshift column
    catalogueData = np.reshape(np.array(hdulData.field(dataCols[0]), dtype=np.float32), [len(hdulData.field(dataCols[0])),1])
    #Add the columns required for the test
    for i in range(1, len(dataCols)):
        catalogueData = np.hstack([catalogueData,np.reshape(np.array(hdulData.field(dataCols[i]), dtype=np.float32), [len(hdulData.field(dataCols[i])),1])])
    fieldList = np.reshape(np.array(hdulData.field("field"), dtype=np.str), [len(hdulData.field("field")),1])
    # print("Time taken to create catalogueData: " + str(datetime.now() - startTime))

    catalogueData1 = pd.DataFrame(catalogueData)
    catalogueData1.to_csv("catalogueData1.csv", index=False)
    #Begin cleaning process
    #Remove items with missing redshifts
    missingRedshifts = np.where(catalogueData[:,0] <= 0)[0]
    cleanCatalogue = np.delete(catalogueData, missingRedshifts, 0)
    fieldList = np.delete(fieldList, missingRedshifts, 0)

    #Make sure values are all within "sane" ranges
    for i in range(1, len(dataCols)):
        cleanCatalogue[:,i] = preprocess.cleanData(cleanCatalogue[:,i], dataType[i], fillMethod)

    cleanCatalogue1 = pd.DataFrame(cleanCatalogue)
    cleanCatalogue1.to_csv("cleanCatalogue1.csv", index=False)


    #print("updated Categologe", cleanCatalogue.shape)
    #Removing min and max redshifts if set
    if minRedshift != None:
        killRedshift = np.where(cleanCatalogue[:,0] < minRedshift)[0]
        cleanCatalogue = np.delete(cleanCatalogue, killRedshift, 0)
    if maxRedshift != None:
        killRedshift = np.where(cleanCatalogue[:,0] > maxRedshift)[0]
        cleanCatalogue = np.delete(cleanCatalogue, killRedshift, 0)
    
    if postBin and not classification:
        temp = cleanCatalogue[:,0]
        temp, binEdges, binnedZ = binDataFunc(temp, binData)
    
    #Bin z values
    if preBin or classification:
        cleanCatalogue[:,0], binEdges, binnedZ = binDataFunc(cleanCatalogue[:,0], binData)
    
    #Take log(z)
    if logZ:
        cleanCatalogue[:,0] = np.log(cleanCatalogue[:,0])
    
    #Use log(Radio)
    if useLogRadio:
        cleanCatalogue[:,1] = np.log(cleanCatalogue[:,1])

    #Use Optical Colours
    if useColoursOptical:
        for i in range(-4, -2):
            cleanCatalogue[:,i] = cleanCatalogue[:,i] - cleanCatalogue[:,i+1]
        cleanCatalogue = cleanCatalogue[:,0:-1]

    #Use IR Colours. Each dataType has different column numbers for IR, hence need different solution to each.
    #Need to take log(IR Flux) to get them into "Magnitudes", which can then be used to calculate the difference
    #between each - "Colours". Given we lose a column of data going to colours, delete the last column.
    #Set useMagnitudes to False so we don't then take a log of a ratio of a log.
    if useColoursIR:
        if colsType == 0:
            for i in range(4,6):
                cleanCatalogue[:,i] = np.log(cleanCatalogue[:,i])
            for i in range(4,5):
                cleanCatalogue[:,i] = cleanCatalogue[:,i] - cleanCatalogue[:,i+1]
            cleanCatalogue = np.delete(cleanCatalogue, obj = 6, axis = 1)
        elif colsType == 1:
            for i in range(3,5):
                cleanCatalogue[:,i] = np.log(cleanCatalogue[:,i])
            for i in range(3,4):
                cleanCatalogue[:,i] = cleanCatalogue[:,i] - cleanCatalogue[:,i+1]
            cleanCatalogue = np.delete(cleanCatalogue, obj = 5, axis = 1)
        elif colsType == 2:
            for i in range(3,4):
                cleanCatalogue[:,i] = np.log(cleanCatalogue[:,i])
            cleanCatalogue[:,2] = cleanCatalogue[:,2] - cleanCatalogue[:,3]
            cleanCatalogue = np.delete(cleanCatalogue, obj = 3, axis = 1)
        useIRMagnitudes = False

    #Take the log(IR Flux) to get the IR "Magnitudes"
    if useIRMagnitudes:
        if colsType == 0:
            for i in range(4,6):
                cleanCatalogue[:,i] = np.log(cleanCatalogue[:,i])
        elif colsType == 1:
            for i in range(3,5):
                cleanCatalogue[:,i] = np.log(cleanCatalogue[:,i])
        elif colsType == 2:
            for i in range(3,4):
                cleanCatalogue[:,i] = np.log(cleanCatalogue[:,i])


    cleanCatalogue1 = pd.DataFrame(cleanCatalogue)
    cleanCatalogue1.to_csv("cleanCatalogue1.csv", index=False)
    print("cleaning done")

    #Standardising all xVals - (x_i - x_mean) / x_sd
    if standardiseXVals:
        for i in range(1, cleanCatalogue.shape[1]):
            cleanCatalogue[:,i] = (cleanCatalogue[:,i] - np.mean(cleanCatalogue[:,i])) / np.std(cleanCatalogue[:,i])

    # print("Time taken to clean and pre-process cleanCatalogue: " + str(datetime.now() - startTime))


    y_vals = cleanCatalogue[:,[0]]
    x_vals = cleanCatalogue[:,1:]
    num_features = x_vals.shape[1]
    predictionBootstrap = []
    mseBootstrap = []
    outlierBootstrap = []
    
    # Split the data into train and test sets
    if testType == 0: 
        #Withdraw our 30% test set
        np.random.seed(225)
        test_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.3), replace=False)
        train_indices = np.array(list(set(range(len(x_vals))) - set(test_indices)))
        x_vals_train = x_vals[train_indices]
        x_vals_test = x_vals[test_indices]
        y_vals_train = y_vals[train_indices]
        y_vals_test = y_vals[test_indices]

    elif testType == 1:
        #Withdraw our test set
        x_vals_test = x_vals[np.where(fieldList == "CDFS    ")[0]]
        y_vals_test = y_vals[np.where(fieldList == "CDFS    ")[0]]
        #Find the training set
        x_vals_train = x_vals[np.where(fieldList == "ELAIS-S1")[0]]
        y_vals_train = y_vals[np.where(fieldList == "ELAIS-S1")[0]]
        
        
    elif testType == 2:
        #Withdraw our test set
        y_vals_test = y_vals[np.where(fieldList == "ELAIS-S1")[0]]
        x_vals_test = x_vals[np.where(fieldList == "ELAIS-S1")[0]]
        #Find the training set
        x_vals_train = x_vals[np.where(fieldList == "CDFS    ")[0]]
        y_vals_train = y_vals[np.where(fieldList == "CDFS    ")[0]]

    print("start")
    x_vals_train_df = pd.DataFrame(x_vals_train)
    y_vals_train_df = pd.DataFrame(y_vals_train)
    vals_train_df = pd.concat([y_vals_train_df, x_vals_train_df], ignore_index=True, axis=1)
    vals_train_df.to_csv('vals_train_df_test_type{dbname}.csv'.format(dbname=testType), index=False)


    print("train size", vals_train_df.shape)
    x_vals_test_df = pd.DataFrame(x_vals_test)
    y_vals_test_df = pd.DataFrame(y_vals_test)
    vals_test_df = pd.concat([y_vals_test_df, x_vals_test_df], ignore_index=True, axis=1)
    vals_test_df.to_csv('vals_test_df_test_type{dbname}.csv'.format(dbname=testType), index=False)
    print("test size", vals_test_df.shape)

    method_no = 6

    if method_no == 1:
        imputed_file_name = "KNN_imputated_catalogueData1.csv"
    elif method_no == 2:
        imputed_file_name = "GAN_imputated_catalogueData1.csv"
    elif method_no == 3:
        imputed_file_name = "Mean_imputated_catalogueData1.csv"
    elif method_no == 4:
        imputed_file_name = "Median_imputated_catalogueData1.csv"
    elif method_no == 5:
        imputed_file_name = "MICE_imputated_catalogueData1.csv"
    else:
        imputed_file_name = "None"

    if imputed_file_name != "None":
        if not os.path.isfile(imputed_file_name):
            sourceFolderPath = "Source File Path/Results/"
            destFolderPath = sourceFolderPath + folderpath + "/"
            shutil.move(os.path.join(sourceFolderPath, imputed_file_name), destFolderPath)

        x_vals_test = np.loadtxt(imputed_file_name, delimiter=",", usecols=(range(1, 10)), skiprows=1)
        y_vals_test = np.loadtxt(imputed_file_name, delimiter=",", usecols=(0), skiprows=1)
        y_vals_test = y_vals_test.reshape(-1)
        print("x size is:", x_vals_test.shape)
        print("y size is: ", y_vals_test.shape)
        print("done")

    if not preBin and metricLearn:
        metricLearnModel = metricLearnRegression(x_vals_train, y_vals_train)
        x_vals_train = metricLearnModel.transform(x_vals_train)
        x_vals_test = metricLearnModel.transform(x_vals_test)


    if type(bootstrapSize) == int:
        predictionBootstrap = []
        mseBootstrap = []
        outlierBootstrap = []

        # for i in tqdm(range(bootstrapSize)):
        for i in range(bootstrapSize):
            # Use metric learning if required
            if metricLearn and not classification:
                B = metricLearnRegression(x_vals_train, y_vals_train)
                x_vals_train = B.transform(x_vals_train)
                x_vals_test = B.transform(x_vals_test)

            # Split the data into train and test sets
            # Randomly sample our training set for bootstrapping
            train_indices = np.random.choice(len(y_vals_train), len(y_vals_train), replace=True)
            x_vals_train_bootstrap = x_vals_train[train_indices]
            y_vals_train_bootstrap = y_vals_train[train_indices]

            kFold = KFold(n_splits=nSplits, random_state=10, shuffle=True)
            MSE = []
            Failed = []

            # for numNeighbours in tqdm(neighboursList):
            for numNeighbours in neighboursList:
                mseList = []
                failed = []

                # for trainIndex, testIndex in tqdm(kFold.split(x_vals_train_bootstrap), total=nSplits):
                for trainIndex, testIndex in kFold.split(x_vals_train_bootstrap):
                    x_vals_train_cross = x_vals_train_bootstrap[trainIndex]
                    x_vals_test_cross = x_vals_train_bootstrap[testIndex]
                    y_vals_train_cross = y_vals_train_bootstrap[trainIndex]
                    y_vals_test_cross = y_vals_train_bootstrap[testIndex]

                    if MLMethod == 0:
                        pred, mseTest = kNN(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross, distType)
                    elif MLMethod == 1:
                        pred, mseTest = linRegress(x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
                    elif MLMethod == 2:
                        pred, mseTest = randomForestRegress(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
                    elif MLMethod == 3:
                        pred, mseTest = lassoRegress(x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
                    elif MLMethod == 4:
                        pred, mseTest = ridgeRegress(x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)

                    lengthOfSplit = len(pred)
                    if logZ:
                        error = np.abs(np.exp(pred) - np.exp(y_vals_test_cross))
                        failed.append(len(error[np.where(error > (FailureLimit * (1+np.exp(y_vals_test_cross))))[0]])/lengthOfSplit )
                    else:
                        error = np.abs(pred - y_vals_test_cross)
                        failed.append(len(error[np.where(error > (FailureLimit * (1+y_vals_test_cross)))[0]])/lengthOfSplit )

                    mseList.append(np.round(mseTest,3))

                MSE.append(np.mean(mseList))
                Failed.append(np.mean(failed))
            
            mseBootstrap.append(MSE)
            outlierBootstrap.append(Failed)

            bestKIndex = (np.argmin(np.array(Failed)))
            bestK = neighboursList[bestKIndex]

            if MLMethod == 0:
                pred, mse_test = kNN(numNeighbours, x_vals_train_bootstrap, x_vals_test, y_vals_train_bootstrap, y_vals_test, distType)
            elif MLMethod == 1:
                pred, mse_test = linRegress(x_vals_train_bootstrap, x_vals_test, y_vals_train_bootstrap, y_vals_test)
            elif MLMethod == 2:
                pred, mse_test = randomForestRegress(numNeighbours, x_vals_train_bootstrap, x_vals_test, y_vals_train_bootstrap, y_vals_test)
            elif MLMethod == 3:
                pred, mse_test = lassoRegress(x_vals_train_bootstrap, x_vals_test, y_vals_train_bootstrap, y_vals_test)
            elif MLMethod == 4:
                pred, mse_test = ridgeRegress(x_vals_train_bootstrap, x_vals_test, y_vals_train_bootstrap, y_vals_test)
            
            if logZ:
                error = np.abs(np.exp(pred) - np.exp(y_vals_test))
                testError = (len(error[np.where(error > (FailureLimit*(1+np.exp(y_vals_test))))[0]])/len(pred) )
            else:
                error = np.abs(pred - y_vals_test)
                testError = (len(error[np.where(error > (FailureLimit*(1+y_vals_test)))[0]])/len(pred) )


            if logZ:
                predictionBootstrap.append(np.exp(pred))
            else:
                predictionBootstrap.append(pred)


    

    outlier_final = []
    mse_final = []
    kFold = KFold(n_splits=nSplits, random_state=10, shuffle=True)
    # for numNeighbours in tqdm(neighboursList):
    for numNeighbours in neighboursList:
        mseList = []
        failed = []
        # TODO: Need to turn this back on when regression metric learn is working.
        if metricLearn and preBin: # and classification:
            lmnn = LMNN(n_neighbors=numNeighbours, max_iter=200, n_features_out=x_vals_train.shape[1], verbose=None)
        
        # for trainIndex, testIndex in tqdm(kFold.split(x_vals_train), total=nSplits):
        for trainIndex, testIndex in kFold.split(x_vals_train):
            #Define training and test sets
            x_vals_train_cross = x_vals_train[trainIndex]
            x_vals_test_cross = x_vals_train[testIndex]
            y_vals_train_cross = y_vals_train[trainIndex]
            y_vals_test_cross = y_vals_train[testIndex]

            # TODO: Need to turn this back on when regression metric learn is working.
            if metricLearn and preBin: # and classification:
                lmnn.fit(x_vals_train_cross, np.squeeze(y_vals_train_cross.astype(str)))
                x_vals_train_cross = lmnn.transform(x_vals_train_cross)
                x_vals_test_cross = lmnn.transform(x_vals_test_cross)
            
            # Use metric learning if required
            if metricLearn and not classification:
                B = metricLearnRegression(x_vals_train_cross, y_vals_train_cross)
                x_vals_train_cross = B.transform(x_vals_train_cross)
                x_vals_test_cross = B.transform(x_vals_test_cross)

            if classification:
                if MLMethod == 0:
                    pred, mseTest = kNN_classification(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross, distType)
                elif MLMethod == 1:
                    pred, mseTest = logRegress(x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
                elif MLMethod == 2:
                    pred, mseTest = randomForestClass(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
            else:
                if MLMethod == 0:
                    pred, mseTest = kNN(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross, distType)
                elif MLMethod == 1:
                    pred, mseTest = linRegress(x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
                elif MLMethod == 2:
                    pred, mseTest = randomForestRegress(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
                elif MLMethod == 3:
                    pred, mseTest = lassoRegress(x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
                elif MLMethod == 4:
                    pred, mseTest = ridgeRegress(x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
                
            lengthOfSplit = len(pred)
            if logZ:
                error = np.abs(np.exp(pred) - np.exp(np.squeeze(y_vals_test_cross)))
                failed.append(len(error[np.where(error > (FailureLimit * (1+np.exp(np.squeeze(y_vals_test_cross)))))[0]])/lengthOfSplit )
            else:
                error = np.abs(pred - np.squeeze(y_vals_test_cross))
                failed.append(len(error[np.where(error > (FailureLimit * (1+np.squeeze(y_vals_test_cross))))[0]])/lengthOfSplit )
        
            mseList.append(np.round(mseTest,3))

        mse_final.append(np.mean(mseList))
        outlier_final.append(np.mean(failed))

    bestKIndex = (np.argmin(np.array(outlier_final)))
    bestK = neighboursList[bestKIndex]

    if classification:
        if metricLearn:
            lmnn = LMNN(n_neighbors=bestK, max_iter=200, n_features_out=x_vals_train.shape[1], verbose=None)
            lmnn.fit(x_vals_train, np.squeeze(y_vals_train.astype(str)))
            x_vals_train = lmnn.transform(x_vals_train)
            x_vals_test = lmnn.transform(x_vals_test)
            
        if MLMethod == 0:
            finalPrediction, finalMSE = kNN_classification(bestK, x_vals_train, x_vals_test, y_vals_train, y_vals_test, distType)
        if MLMethod == 1:
            finalPrediction, finalMSE = logRegress(x_vals_train, x_vals_test, y_vals_train, y_vals_test)
        if MLMethod == 2:
            finalPrediction, finalMSE = randomForestClass(bestK, x_vals_train, x_vals_test, y_vals_train, y_vals_test)

        
    else:
        # TODO: Need to remove/change this once regression metric learning is done.
        if metricLearn and preBin: 
            lmnn = LMNN(n_neighbors=bestK, max_iter=200, n_features_out=x_vals_train.shape[1], verbose=None)
            lmnn.fit(x_vals_train, np.squeeze(y_vals_train.astype(str)))
            x_vals_train = lmnn.transform(x_vals_train)
            x_vals_test = lmnn.transform(x_vals_test)

        if MLMethod == 0:
            finalPrediction, finalMSE = kNN(bestK, x_vals_train, x_vals_test, y_vals_train, y_vals_test, distType)
        elif MLMethod == 1:
            finalPrediction, finalMSE = linRegress(x_vals_train, x_vals_test, y_vals_train, y_vals_test)
        elif MLMethod == 2:
            finalPrediction, finalMSE = randomForestRegress(bestK, x_vals_train, x_vals_test, y_vals_train, y_vals_test)
        elif MLMethod == 3:
            finalPrediction, finalMSE = lassoRegress(x_vals_train, x_vals_test, y_vals_train, y_vals_test)
        elif MLMethod == 4:
            finalPrediction, finalMSE = ridgeRegress(x_vals_train, x_vals_test, y_vals_train, y_vals_test)

    residuals = (np.squeeze(y_vals_test) - finalPrediction) / (1 + np.squeeze(y_vals_test))

    if postBin:
        finalPrediction, temp, temp2 = binDataFunc(finalPrediction, binData, binEdges = binEdges, newZ = binnedZ)
        y_vals_test, temp, temp2 = binDataFunc(y_vals_test, binData, binEdges = binEdges, newZ = binnedZ)
        
    if postBin or classification:
        confusion = confusion_matrix(np.round(y_vals_test,2).astype(str),np.round(finalPrediction,2).astype(str))#.astype(float)        
        #plotNormConfusionMatrix(confusion,binnedZ,binEdges)
        plotScaledConfusionMatrix(y_vals_test, finalPrediction, binEdges, binnedZ)
        mutualInfo = adjusted_mutual_info_score(np.squeeze(y_vals_test).astype(str),np.squeeze(finalPrediction).astype(str))



    if logZ:
        error = np.abs(np.exp(finalPrediction) - np.exp(np.squeeze(y_vals_test)))
        testError = (len(error[np.where(error > (FailureLimit*(1+np.exp(np.squeeze(y_vals_test)))))[0]])/len(finalPrediction) )
    else:
        error = np.abs(finalPrediction - np.squeeze(y_vals_test))
        testError = (len(error[np.where(error > (FailureLimit*(1+np.squeeze(y_vals_test))))[0]])/len(finalPrediction) )




    if type(bootstrapSize) == int:
        percentiles = np.percentile(predictionBootstrap, q=[2.5,97.5], axis=0)
        percentiles[0,:] = np.abs(finalPrediction -  percentiles[0,:])
        percentiles[1,:] = np.abs(percentiles[1,:] - finalPrediction)



    if classification or postBin:
        precision = metrics.precision_score(y_vals_test.ravel().astype(str), finalPrediction.ravel().astype(str), average="macro")
        recall = metrics.recall_score(y_vals_test.ravel().astype(str), finalPrediction.ravel().astype(str), average="macro")
        f1 = metrics.f1_score(y_vals_test.ravel().astype(str), finalPrediction.ravel().astype(str), average="macro")
    else:
        mse = metrics.mean_squared_error(y_vals_test.ravel(), finalPrediction.ravel())

    predFile = "finalPredictions"
    yValsFile = "yValsFile"
    mseFile = "mseFile"
    outlierFile = "outlierFile"
    binEdgesFile = "binEdges"

    with open(predFile, "wb") as openFile:
        pickle.dump(finalPrediction, openFile)

    with open(yValsFile, "wb") as openFile:
        pickle.dump(y_vals_test, openFile)

    with open(mseFile, "wb") as openFile:
        pickle.dump(mse_final, openFile)

    with open(outlierFile, "wb") as openFile:
        pickle.dump(outlier_final, openFile)

    if postBin or classification:
        with open(binEdgesFile, "wb") as openFile:
            pickle.dump(binEdges, openFile)
    
    outlierRate = 100*len(residuals[np.where(abs(residuals)>0.15)])/len(residuals)

    with open("results.csv", "w") as openFile:
        if postBin or classification:
            openFile.write("bestK,numTrainSources,numTestSources,outlier,score,mutualInfo,residual_std_dev,precision,recall,f1,time\n")
            openFile.write(str(bestK) + "," + str(y_vals_train.shape[0]) + "," + str(y_vals_test.shape[0]) + "," + str(outlierRate) + "," + str(finalMSE) + "," + str(mutualInfo) + "," + str(np.std(residuals)) + "," + str(precision) + "," + str(recall) + "," + str(f1) + "," + str(datetime.now() - startTime))
        else:
            openFile.write("bestK,numTrainSources,numTestSources,outlier,score,residual_std_dev,mse,time\n")
            openFile.write(str(bestK) + "," + str(y_vals_train.shape[0]) + "," + str(y_vals_test.shape[0]) + "," + str(outlierRate) + "," + str(finalMSE) + "," + str(np.std(residuals)) + "," + str(mse) + "," + str(datetime.now() - startTime))

    #Find number of test sources to use in the plot titles

    if MLMethod != 3 and MLMethod != 4 and MLMethod != 1:
        plt.figure(0)
        if classification:
            plt.plot(neighboursList, np.array(mse_final), color="springgreen", label="Accuracy")
        else:
            plt.plot(neighboursList, np.array(mse_final), color="springgreen", label=r'R$^2$')
        plt.plot(neighboursList, np.array(outlier_final), color="deepskyblue", label="Failure Rate")
        plt.ylabel("Error Metric")
        if MLMethod == 0:
            plt.xlabel('Number of Neighbours')
        elif MLMethod == 2:
            plt.xlabel("Number of Trees")
        plt.axvline(bestK,color="red", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig("cross_validation.pdf")



    if logZ:
        y_vals_test = np.exp(y_vals_test)



    if type(bootstrapSize) == bool:
        plotData(np.squeeze(y_vals_test), finalPrediction, plt, stats, pylab)
    else:
        plotData(np.squeeze(y_vals_test), finalPrediction, plt, stats, pylab, percentiles)


    plt.savefig("resultsPlot.pdf")

if __name__ == "__main__":
	main()
