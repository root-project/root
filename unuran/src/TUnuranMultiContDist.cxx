// @(#)root/unuran:$Name:  $:$Id: TUnuranMultiContDist.cxx,v 1.1 2007/03/08 09:31:54 moneta Exp $
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranMultiContDist

#include "TUnuranMultiContDist.h"

#include "TF1.h"
#include <cassert>


TUnuranMultiContDist::TUnuranMultiContDist (TF1 * func, unsigned int dim, bool isLogPdf) : 
   fPdf(func), 
   fDim( dim ),
   fIsLogPdf(isLogPdf)
{
   //Constructor from a TF1 objects
   if (fDim == 0) fDim = func->GetNdim(); 
} 



TUnuranMultiContDist::TUnuranMultiContDist(const TUnuranMultiContDist & rhs) : 
   TUnuranBaseDist()
{
   // Implementation of copy ctor using assignment operator
   operator=(rhs);
}

TUnuranMultiContDist & TUnuranMultiContDist::operator = (const TUnuranMultiContDist &rhs) 
{
   // Implementation of assignment operator (copy only the funciton pointer not the function itself)
   if (this == &rhs) return *this;  // time saving self-test
   fPdf  = rhs.fPdf;
   fDim  = rhs.fDim;
   fXmin = rhs.fXmin;
   fXmax = rhs.fXmax;
   fMode = rhs.fMode;
   fIsLogPdf  = rhs.fIsLogPdf;
   return *this;
}



double TUnuranMultiContDist::Pdf ( const double * x) const {  
   // evaluate the distribution 
   assert(fPdf != 0);
   return fPdf->EvalPar(x); 
}


void TUnuranMultiContDist::Gradient( const double * x, double * grad) const { 
      // do numerical derivation of gradient
   std::vector<double> g(fDim); 
   for (unsigned int i = 0; i < fDim; ++i) 
      g[i] = Derivative(x,i); 
      
   grad = &g.front();
   return;
}

double TUnuranMultiContDist::Derivative( const double * x, int coord) const { 
    // do numerical derivation of gradient using 5 point rule
   // use 5 point rule 

   //double eps = 0.001; 
   //const double kC1 = 8*std::numeric_limits<double>::epsilon();
   assert(fPdf != 0);

   double h = 0.001; 

   std::vector<double> xx(fDim);
   double * params = fPdf->GetParameters();
   fPdf->InitArgs(&xx.front(), params);
 
   xx[coord] = x[coord]+h;     double f1 = fPdf->EvalPar(&xx.front(),params);
   //xx[coord] = x[coord];       double fx = fPdf->EvalPar(&xx.front(),params);
   xx[coord] = x[coord]-h;     double f2 = fPdf->EvalPar(&xx.front(),params);

   xx[coord] = x[coord]+h/2;   double g1 = fPdf->EvalPar(&xx.front(),params);
   xx[coord] = x[coord]-h/2;   double g2 = fPdf->EvalPar(&xx.front(),params);

   //compute the central differences
   double h2    = 1/(2.*h);
   double d0    = f1 - f2;
   double d2    = 2*(g1 - g2);
   //double error  = kC1*h2*fx;  //compute the error
   double deriv = h2*(4*d2 - d0)/3.;  
   return deriv;
}



