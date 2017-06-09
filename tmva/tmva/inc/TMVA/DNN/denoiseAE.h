#ifndef DAELAYER_H_
#define DAELAYER_H_

#include <cstdlib>
#include <cmath>
namespace TMVA
{
/*simple denoising Autoencoder

  It is an unsupervised learning algorithm (like PCA)
  It minimizes the same objective function as PCA
  It is a neural network
  The neural network’s target output is its input
*/

class DAE
{
public:

  int n;
  //number of visible units
  int numVisible;
  //number of hidden units
  int numHidden;
  //weight matrix
  double **weight;
  //bias for hidden units
  double *hiddenBias;
  //bias for visible units
  double *visibleBias;
  //constructor for class DAE
  DAE(int size, int visibleUnits, int hiddenUnits, double** weights, double* hiddenBiasValue, double* visibleBiasUnits);
  //destructor
  ~DAE();
  //function that takes input x and add noise to the input
  void corruptInput(int* x, int* tildeX, double corruptionLevel);
  //function that takes an input x  and generates y by simple forward prop
  void encodeHiddenValues(int* x, double* y);
  //function that reconstruct the matrix z ~ x
  void reconstructInput(double* y, double* z);
  void train(int* x, double learningRate, double corruptionLevel);
  void reconstruct(int*, double*);
  double getWeights();
  double randInit (double min, double max);
  int binomial (int n, double p);
  double sigmoid(double x);
};
/*
________________________________________________________________________________
// implementation of randInit for initialializing random values
________________________________________________________________________________
*/
double DAE:: randInit(double min, double max)
{
  return rand() / (RAND_MAX + 1.0) * (max - min) + min ;
}

double DAE:: getWeights()
{
  for(int i=0;i<numHidden;i++)
  {
    for(int j=0; j<numVisible; j++)
    {
      std::cout<<weight[i][j];

    }
  }
}
//______________________________________________________________________________
// implementation of binomial or adding noise to i/p
//______________________________________________________________________________
int DAE:: binomial(int n, double p)
{
  if(p < 0 || p > 1) return 0;
  int c = 0;
  double r;
  for(int i=0; i<n; i++)
  {
    r = rand() / (RAND_MAX + 1.0);
    if (r < p) c++;
  }
  return c;
}
//______________________________________________________________________________
//  implementation of sigmoid fn.
//______________________________________________________________________________
double DAE:: sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}
//______________________________________________________________________________

//______________________________________________________________________________

/*
This function keeps ``1-corruption_level`` entries of the inputs the
same and zero-out randomly selected subset.
this will produce an array of 0s and 1s where 1 has a
probability of 1 - ``corruption_level`` and 0 with
``corruption_level``
all we need to do is to add a stochastic corruption step operating on the
input. The input can be corrupted in many ways, but here we will
stick to the original corruption mechanism of randomly masking entries of the
input by making them zero.

** params x                   input
** params tildeX             to get input with some noise
** params corruption_level    a corruption level factor

*/

void DAE::corruptInput(int* x, int* tildeX, double corruptionLevel)
{
  for(int i=0; i<numVisible; i++)
  {
    if (x[i]==0)
    {
      tildeX[i]=0;
    }
    else
    {
    tildeX[i] = binomial(1,corruptionLevel);
    }
    tildeX[i] = x[i];
  }
  //for (int i=0; i<numVisible; i++)
  //{
    //std::cout<< tildeX[i];
  //}

}
//______________________________________________________________________________

//______________________________________________________________________________

//encode the hidden values, simple forward prop step to get output y
//Wx+b
void DAE::encodeHiddenValues(int *x, double *y)
{
  for(int i = 0; i < numHidden; i++)
  {
    y[i] = 0;
    for(int j = 0; j < numVisible; j++)
    {
      y[i] += weight[i][j]*x[j];
    }
    y[i] += hiddenBias[i];
    y[i] = sigmoid(y[i]);
  }
}


//______________________________________________________________________________

//______________________________________________________________________________
//decode the hidden values to generate z
//z ~ x
void DAE::reconstructInput(double *y, double *z)
{
  for(int i=0; i<numVisible; i++)
  {
    z[i] = 0;
    for(int j=0; j<numHidden; j++)
    {
      z[i] += weight[j][i]*y[j];
    }
    z[i] += visibleBias[i];
    z[i] = sigmoid(z[i]);
  }
}

//______________________________________________________________________________

//______________________________________________________________________________

}//TMVA
#endif
