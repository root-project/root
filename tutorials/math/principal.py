## \file
## \ingroup tutorial_math
## \notebook
## Principal Components Analysis (PCA) example
##
## Example of using TPrincipal as a stand alone class.
##
## I create n-dimensional data points, where c = trunc(n / 5) + 1
## are  correlated with the rest n - c randomly distributed variables.
##
## Based on principal.C by Rene Brun and Christian Holm Christensen
##
## \macro_output
## \macro_code
##
## \authors Juan Fernando, Jaramillo Botero

from ROOT import TPrincipal, gRandom, TBrowser, vector


n = 10
m = 10000

c = int(n / 5) + 1

print ("""*************************************************
*         Principal Component Analysis          *
*                                               *
*  Number of variables:           {0:4d}          *
*  Number of data points:         {1:8d}      *
*  Number of dependent variables: {2:4d}          *
*                                               *
*************************************************""".format(n, m, c))

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
data = vector('double')()
for i in range(m):
    # First we create the un-correlated, random variables, according
    # to one of three distributions
    for j in range(n - c):
        if j % 3 == 0:
            data.push_back(randumNum.Gaus(5, 1))
        elif j % 3 == 1:
            data.push_back(randumNum.Poisson(8))
        else:
            data.push_back(randumNum.Exp(2))

    # Then we create the correlated variables
    for j in range(c):
        data.push_back(0)
        for k in range(n - c - j):
            data[n - c + j] += data[k]

    # Finally we're ready to add this datapoint to the PCA
    principal.AddRow(data.data())
    data.clear()

# Do the actual analysis
principal.MakePrincipals()

# Print out the result on
principal.Print()

# Test the PCA
principal.Test()

# Make some histograms of the orginal, principal, residue, etc data
principal.MakeHistograms()

# Make two functions to map between feature and pattern space
# Start a browser, so that we may browse the histograms generated
# above
principal.MakeCode()
b = TBrowser("principalBrowser", principal)
