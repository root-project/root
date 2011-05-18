// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class FitUtil


#ifdef _WIN32  // no support for parallel fits on Win32 
#ifdef ROOT_FIT_PARALLEL
#undef ROOT_FIT_PARALLEL
#endif
#endif

#ifdef ROOT_FIT_PARALLEL

#ifndef ROOT_Fit_FitUtilParallel
#define ROOT_Fit_FitUtilParallel

#ifndef ROOT_Math_IParamFunctionfwd
#include "Math/IParamFunctionfwd.h"
#endif

#ifndef ROOT_Fit_DataVectorfwd
#include "Fit/DataVectorfwd.h"
#endif


namespace ROOT { 

   namespace Fit { 


   typedef  ROOT::Math::IParamMultiFunction IModelFunction;
   

/** 
   namespace defining free functions for Fitting parallel mode 

   Status: functions are not not completed and are still preliminary

   @ingroup FitMain
*/ 
namespace FitUtilParallel {



   /** 
       evaluate the Chi2 given a model function and the data at the point x. 
       return also nPoints as the effective number of used points in the Chi2 evaluation
       Use a parallel evaluation spawning multiple threads 
   */ 
   double EvaluateChi2(IModelFunction & func, const BinData & data, const double * x, unsigned int & nPoints);  


   double EvaluateLogL(IModelFunction & func, const UnBinData & data, const double * p, unsigned int &nPoints);  


} // end namespace FitUtil 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_FitUtil */
#endif /* ROOT_FIT_PARALLEL */
