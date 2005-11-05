// @(#)root/minuit2:$Name:  $:$Id: TFcnAdapter.cxx,v 1.1 2005/10/27 14:11:07 brun Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TFcnAdapter.h"

#include <cassert>
//#include <iostream>

double TFcnAdapter::operator()(const std::vector<double>& par) const {
//   assert(par.size() == theNPar);
//   std::cout<<"TFcnAdapter::operator()"<<std::endl;
  assert(fFCN != 0);
  double fs = 0.;
//   double* theCache = new double[par.size()];
//   copy(par.begin(), par.end(), theCache);

  double* theCache = (double*)(&(par.front()));

  int npar = par.size();
  (*fFCN)(npar, 0, fs, theCache, 4);
//   delete [] theCache;

  return fs;
}
  
std::vector<double> TFcnAdapter::gradient(const std::vector<double>& par) const {
//     std::cout<<"TFcnAdapter::gradient "<<std::endl;
//   assert(par.size() == theNPar);
  assert(fFCN != 0);
  double fs = 0.;
  int npar = par.size();
  double* theCache = new double[par.size()];
  double* theGradCache = new double[par.size()];
  for(int i = 0; i < npar; i++) theCache[i] = par[i];
//   (*theFcn)(npar, theGradCache, fs, theCache, 2);
  (*fFCN)(npar, theGradCache, fs, theCache, 4);
  std::vector<double> grad(theGradCache, theGradCache+npar);
  return std::vector<double>(theGradCache, theGradCache+npar);
}

// forward interface
double TFcnAdapter::operator()(int npar, double* params) const {
//   std::cout<<"TFcnAdapter::operator()(int npar,"<<std::endl;
//   assert(npar == int(theNPar));
  assert(fFCN != 0);
  double fs = 0.;
  (*fFCN)(npar, 0, fs, params, 4);
  return fs;
}
