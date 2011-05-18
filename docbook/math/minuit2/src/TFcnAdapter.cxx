// @(#)root/minuit2:$Id$
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
   // adapt ROOT FCN interface to be called by Minuit2 (to have a FCNBase signature)
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

std::vector<double> TFcnAdapter::Gradient(const std::vector<double>& par) const {
   // adapt ROOT FCN interface (for gradient) to be called by Minuit2 (to have a FCNBase signature)   

   //     std::cout<<"TFcnAdapter::gradient "<<std::endl;
   //   assert(par.size() == theNPar);
   assert(fFCN != 0);
   double fs = 0.;
   int npar = par.size();

   double* theCache = (double*)(&(par.front()));
   if (fGradCache.size() != par.size() ) 
      fGradCache = std::vector<double>(par.size() );

   for(int i = 0; i < npar; i++) theCache[i] = par[i];
   //   (*theFcn)(npar, theGradCache, fs, theCache, 2);
   (*fFCN)(npar, &fGradCache[0], fs, theCache, 4);
   return fGradCache;
}


double TFcnAdapter::operator()(int npar, double* params,int iflag) const {
   // interface using double * instead of std::vector 
   
   //   std::cout<<"TFcnAdapter::operator()(int npar,"<<std::endl;
   //   assert(npar == int(theNPar));
   assert(fFCN != 0);
   double fs = 0.;
   (*fFCN)(npar, 0, fs, params, iflag);
   return fs;
}

