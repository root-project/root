// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TF1.h"
#include <vector>
#include <cassert>

/// utility functions to be used in the fitter classes 

namespace FitterUtil { 
   
   
   double EvalIntegral(TF1 * func, const std::vector<double> & x1, const std::vector<double> & x2, const std::vector<double> & par) {  
      // evaluate integral of fit functions from x1 and x2 and divide by dx
      
      double fval;
      unsigned int ndim = x1.size();
      double dx = x2[0]-x1[0];
      assert (dx != 0);
      if ( ndim == 1) { 
         fval =  func->Integral( x1[0],x2[0], &par.front() )/dx;
         return fval;
      }
      // dim > 1
      double dy = x2[1]-x1[1];
      assert (dy != 0);
      func->SetParameters(&par.front() );
      if ( ndim == 2) { 
         fval = func->Integral( x1[0],x2[0],x1[1],x2[1] )/(dx*dy);
         return fval;
      }
      // dim = 3 
      double dz = x2[2]-x1[2];
      assert (dz != 0);
      fval = func->Integral( x1[0],x2[0],x1[1],x2[1],x1[2],x2[2])/(dx*dy*dz);
      return fval;
      
   }



} // end namespace FitterUtil
