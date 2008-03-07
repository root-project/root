// @(#)root/fit:$Id$
// Author: L. Moneta Wed Aug 30 11:04:59 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class DataOptions

#ifndef ROOT_Fit_DataOptions
#define ROOT_Fit_DataOptions


namespace ROOT { 

   namespace Fit { 


/** 
   DataOptions : simple structure holding the options on how the data are filled 
*/ 
struct DataOptions {


   /** 
      Default constructor: have default options
   */ 
   DataOptions () : 
      fIntegral(false), 
      fUseEmpty(false), 
      fUseRange(false), 
      fErrors1(false),
      fCoordErrors(false),
      fAsymErrors(false)
   {}


   bool fIntegral; 
   bool fUseEmpty; 
   bool fUseRange; 
   bool fErrors1;      // use all errors equal to 1 (fit without errors)
   bool fCoordErrors; // use errors on the coordinates when available 
   bool fAsymErrors;  // use asymmetric errors in the value when available (depending on sign of residual)


}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_DataOptions */
