// @(#)root/hist:$Name:  $:$Id: TF1.cxx,v 1.138 2007/05/28 14:35:35 brun Exp $
// Author: Lorenzo Moneta 12/06/07

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// helper functions used internally by TF1

#ifndef ROOT_TF1Helper
#define ROOT_TF1Helper 

namespace ROOT { 

   namespace TF1Helper { 

      double IntegralError(TF1 * func, double a, double b, double eps); 

   } // end namespace TF1Helper

} // end namespace TF1

#endif
