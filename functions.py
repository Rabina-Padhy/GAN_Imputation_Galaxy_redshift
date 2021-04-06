import numpy as np

class DataProcessing(object):

    def __init__(self,boundsList = [[-1,20],[0,1e5], [-10,10], [0,1e6], [10,26], [0,1]]):
        """Initialiser to store the stuff correctly
        
        Default Arrangement of bounds:
         - Redshift - -1 --> 20
         - Radio Flux density - 0 -> 1e5
         - Radio Spectral Index - -10 -> 10
         - Swire (IR) Fluxes - 0 -> 1e6
         - Magnitudes - 10 -> 26
         - Stellarity - 0 -> 1
        Arguments:
            bounds {tuple} -- The bounds for each data type expected - for example Radio Flux, Radio Spectral Index, Optical Magnitude, etc.
        """
        self.__bounds = boundsList


    def cleanData(self,data,dataType, fillMethod):
        """Function to replace the data outside the data type's bounds with the mean of the data within
        
        Arguments:
            data {numpy array} -- Numpy array to be sanatised
            dataType {int} -- Integer index representing the min and max bounds for the data set, related to those used to setup DataProcessing class
            fillMethod {int} -- Integer index representing the method to fill missing values. 0 is replacing missing values with the mean of the column. 1 is the median
        """

        #Create local vals for easy reference
        minBound = self.__bounds[dataType][0]
        maxBound = self.__bounds[dataType][1]

        

        #gotta start somewhere if it's to be simple. All values larger than the max bound are changed to min
        missingValues = np.where(data >= maxBound)
        data[missingValues] = -999
        #Then find all missing vals
        missingValues = np.where(data <= minBound)
        #Calculate mean value based on non-missing values
        if fillMethod == 0:
            replacementValue = np.mean(data[np.where(data > minBound)])
        elif fillMethod == 1:
            replacementValue = np.median(data[np.where(data > minBound)])
        #Assign missing values the mean
        data[missingValues] = replacementValue
        
        return data

    def mad(self, data, axis=None):
        return np.median(np.absolute(data - np.median(data, axis)), axis)

    def plotData(self, specZ, predZ, fileName, plt, stats, pylab, error = False):
        num = specZ.shape[0]
        residual=(specZ-predZ)/(1+specZ)
        plt.rc('text', usetex=True)
        plt.rcParams.update({'font.size': 14})
        
        fig, [ax,ay] = plt.subplots(2, sharex=True,  gridspec_kw = {'height_ratios':[2, 1]})
        fig.set_figheight(9)
        fig.set_figwidth(6)
        sizeElem=2
        if type(error) == bool:
            cax=ax.scatter(specZ, predZ, edgecolor='face', s=sizeElem, cmap="tab20c", color="black", label='Not AGN')
            cay=ay.scatter(specZ, residual, edgecolor='face', s=sizeElem, cmap="tab20c", color="black", label='Not AGN')
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
        nmad=1.4826*self.mad(residual)
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
        pylab.xlabel('$z_{spec}$')
        F = pylab.gcf()
        plt.tight_layout()
        #F.savefig(fileName, dpi = (200))
        pylab.draw()
        plt.show()
        plt.close()
        

    def calcMutualInfo(self,dataTable):
        """Function to calculate the Mutual Information from a confusion matrix.
        
        Arguments:
            dataTable {Numpy Array} -- Confusion matrix stored in a numpy array.
        
        Returns:
            [Float] -- Mutual Information value of the confusion matrix supplied.
        """

        conditionalProb = dataTable/np.sum(dataTable)
        probOfX = np.sum(conditionalProb, axis=1)
        probOfY = np.sum(conditionalProb, axis=0)
        tempTable = (conditionalProb) / (probOfX * probOfY)
        tempTable[tempTable == 0] = 1
        mutualInfo = conditionalProb * np.log(tempTable)
        return np.sum(mutualInfo)

    
    def makeColours(self, vals ):
        from matplotlib.colors import Normalize
        from matplotlib import cm

        colours = np.zeros( (len(vals),3) )
        norm = Normalize( vmin=vals.min(), vmax=vals.max() )

        #Can put any colormap you like here.
        colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

        return colours