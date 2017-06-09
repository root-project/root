#include <iostream>
#include "denoiseAE.h"
using namespace TMVA;

DAE::DAE(int size, int numVisibleUnits, int numHiddenUnits, double **_weight, double* _hiddenBias, double* _visibleBias):
       n(size), numVisible(numVisibleUnits), numHidden(numHiddenUnits)
{

  /*if weights and biases are not yet initialized*/
  if(_weight==NULL)
  {
    weight = new double*[numHidden];
    for(int i=0; i<numHidden; i++)
    {
      weight[i]= new double[numVisible];
    }
    double a = 1.0/ numVisible;
    for(int i=0;i<numHidden;i++)
    {
      for(int j=0; j<numVisible; j++)
      {
        weight[i][j]= randInit(-a,a);

      }
    }
  }
  else
  {
    weight = _weight;
  }



  if(_hiddenBias==NULL)
  {
    hiddenBias = new double[numHidden];
    for(int i=0; i<numHidden; i++)
    {
      hiddenBias[i]=0;
    }
  }
  else
  {
    hiddenBias= _hiddenBias;
  }
  if(_visibleBias==NULL)
  {
    visibleBias= new double[numVisible];
    for(int i=0; i<numVisible; i++)
    {
      visibleBias[i]=0;
    }
  }
  else
  {
    visibleBias=_visibleBias;
  }
}
//destructor
DAE::~DAE()
{
  for(int i=0; i<numHidden; i++)
  {
    delete[] weight[i];
  }
  delete[] weight;
  delete[] hiddenBias;
  delete[] visibleBias;
}



void DAE::train(int *x, double learningRate, double corruptionLevel)
{
  int *tildeX = new int[numVisible];
  double *y = new double[numHidden];
  double *z = new double[numVisible];
  double *visibleError = new double[numVisible];
  double *hiddenError = new double[numHidden];


  corruptInput(x,tildeX,corruptionLevel);
  encodeHiddenValues(tildeX, y);
  reconstructInput(y,z);

  //updating visibleBias
  for(int i=0; i<numVisible; i++)
  {
    visibleError[i] = x[i] - z[i];
    visibleBias[i] += learningRate * visibleError[i] / n;
  }

  //updating hiddenBias
  for(int i=0; i<numHidden; i++)
  {
    hiddenError[i]=0;
    for(int j=0; j<numVisible; j++)
    {
      hiddenError[i] += weight[i][j] * visibleError[j];
    }
    hiddenError[i] *= y[i] * (1-y[i]);
    hiddenBias[i] += learningRate * hiddenError[i]/n;
  }

  //weight
  for(int i=0; i<numHidden;i++)
  {
    for(int j=0; j<numVisible; j++)
    {
      weight[i][j] += learningRate*(hiddenError[i] * tildeX[j] + visibleError[j]*y[i])/n;

    }
  }

  delete[] hiddenError;
  delete[] visibleError;
  delete[] z;
  delete[] y;
  delete[] tildeX;
}

void DAE::reconstruct(int *x, double *z)
{
  double *y = new double[numHidden];
  encodeHiddenValues(x,y);
  reconstructInput(y,z);
  delete[] y;
}
