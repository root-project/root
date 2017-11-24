## \file
##    \ingroup tutorial_math
## \notebook
## Principal Components Analysis (PCA) example
##
## Example of using TPrincipal as a stand alone class.
##
## We create n-dimensional data points, where c = trunc(n / 5) + 1
## are  correlated with the rest n - c randomly distributed variables.
##
## \macro_output
## \macro_code
##
## \authors Juan Fernando Jaramillo Botero
##
## based on principal.C by Rene Brun and Christian Holm Christensen

from ROOT import TPrincipal, gRandom, TBrowser
import numpy


def principal(n=10, m=10000):
    c = n / 5 + 1

    print ("*************************************************")
    print ("*         Principal Component Analysis          *")
    print ("*                                               *")
    print ("*  Number of variables:           {0:4d}          *".format(n))
    print ("*  Number of data points:         {0:8d}      *".format(m))
    print ("*  Number of dependent variables: {0:4d}          *".format(c))
    print ("*                                               *")
    print ("*************************************************")

    # Initilase the TPrincipal object. Use the empty string for the
    # final argument, if you don't wan't the covariance
    # matrix. Normalising the covariance matrix is a good idea if your
    # variables have different orders of magnitude.
    principal = TPrincipal(n, "ND")

    # Use a pseudo-random number generator
    randumNum = gRandom

    # Make the m data-points
    # Make a variable to hold our data
    # Allocate memory for the data point
    data = []
    for i in range(m):
        # First we create the un-correlated, random variables, according
        # to one of three distributions
        for j in range(n - c):
            if j % 3 == 0:
                data.append(randumNum.Gaus(5, 1))
            elif j % 3 == 1:
                data.append(randumNum.Poisson(8))
            else:
                data.append(randumNum.Exp(2))

        # Then we create the correlated variables
        for j in range(c):
            data.append(0)
            for k in range(n - c - j):
                data[n - c + j] += data[k]

        # Finally we're ready to add this datapoint to the PCA
        principal.AddRow(numpy.array(data))

    # Do the actual analysis
    principal.MakePrincipals()

    # Print out the result on
    principal.Print()

    # Test the PCA
    principal.Test()

    # Make some histograms of the orginal, principal, residue, etc data
    principal.MakeHistograms()

    # Make two functions to map between feature and pattern space
    principal.MakeCode()

    # Start a browser, so that we may browse the histograms generated
    # above
    b = TBrowser("principalBrowser", principal)


if __name__ == "__main__":
    principal()
