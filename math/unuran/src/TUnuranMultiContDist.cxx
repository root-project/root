// @(#)root/unuran:$Id$
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranMultiContDist

#include "TUnuranMultiContDist.h"
#include "Math/WrappedMultiTF1.h"

#include "TF1.h"
#include <cassert>


TUnuranMultiContDist::TUnuranMultiContDist (const ROOT::Math::IMultiGenFunction & pdf, bool isLogPdf) : 
   fPdf(&pdf), 
   fIsLogPdf(isLogPdf), 
   fOwnFunc(false)
{
   //Constructor from generic function interfaces
} 


TUnuranMultiContDist::TUnuranMultiContDist (TF1 * func, unsigned int dim, bool isLogPdf) : 
   fPdf(0), 
   fIsLogPdf(isLogPdf), 
   fOwnFunc(false)
{
   //Constructor from a TF1 objects
   if (func) { 
      fPdf = new ROOT::Math::WrappedMultiTF1( *func, dim);
      fOwnFunc = true; 
   }
} 



TUnuranMultiContDist::TUnuranMultiContDist(const TUnuranMultiContDist & rhs) : 
   TUnuranBaseDist(), 
   fPdf(0)
{
   // Implementation of copy ctor using assignment operator
   operator=(rhs);
}

TUnuranMultiContDist & TUnuranMultiContDist::operator = (const TUnuranMultiContDist &rhs) 
{
   // Implementation of assignment operator (copy only the function pointer not the function itself)
   if (this == &rhs) return *this;  // time saving self-test
   fXmin = rhs.fXmin;
   fXmax = rhs.fXmax;
   fMode = rhs.fMode;
   fIsLogPdf  = rhs.fIsLogPdf;
   fOwnFunc   = rhs.fOwnFunc;
   if (!fOwnFunc)  
      fPdf   = rhs.fPdf;
   else { 
       if (fPdf) delete fPdf;
       fPdf  = (rhs.fPdf)  ? rhs.fPdf->Clone()  : 0;  
   }
   return *this;
}

TUnuranMultiContDist::~TUnuranMultiContDist() { 
   // destructor implementation
   if (fOwnFunc && fPdf) delete fPdf; 
}


double TUnuranMultiContDist::Pdf ( const double * x) const {  
   // evaluate the distribution 
   assert(fPdf != 0);
   return (*fPdf)(x); 
}


void TUnuranMultiContDist::Gradient( const double * x, double * grad) const { 
   // do numerical derivation and return gradient in vector grad
   // grad must point to a vector of at least ndim size
   unsigned int ndim = NDim();
   for (unsigned int i = 0; i < ndim; ++i) 
      grad[i] = Derivative(x,i); 
      
   return;
}

double TUnuranMultiContDist::Derivative( const double * x, int coord) const { 
    // do numerical derivation of gradient using 5 point rule
   // use 5 point rule 

   //double eps = 0.001; 
   //const double kC1 = 8*std::numeric_limits<double>::epsilon();
   assert(fPdf != 0);

   double h = 0.001; 

   std::vector<double> xx(NDim() );
 
   xx[coord] = x[coord]+h;     double f1 = (*fPdf)(&xx.front());
   xx[coord] = x[coord]-h;     double f2 = (*fPdf)(&xx.front());

   xx[coord] = x[coord]+h/2;   double g1 = (*fPdf)(&xx.front());
   xx[coord] = x[coord]-h/2;   double g2 = (*fPdf)(&xx.front());

   //compute the central differences
   double h2    = 1/(2.*h);
   double d0    = f1 - f2;
   double d2    = 2*(g1 - g2);
   //double error  = kC1*h2*fx;  //compute the error
   double deriv = h2*(4*d2 - d0)/3.;  
   return deriv;
}



