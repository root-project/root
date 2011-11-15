// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class FitUtil

#ifndef ROOT_Fit_FitUtil
#define ROOT_Fit_FitUtil

#ifndef ROOT_Math_IParamFunctionfwd
#include "Math/IParamFunctionfwd.h"
#endif

#ifndef ROOT_Fit_DataVectorfwd
#include "Fit/DataVectorfwd.h"
#endif


namespace ROOT { 

   namespace Fit { 


   

/** 
   namespace defining utility free functions using in Fit for evaluating the various fit method 
   functions (chi2, likelihood, etc..)  given the data and the model function 

   @ingroup FitMain
*/ 
namespace FitUtil {

   typedef  ROOT::Math::IParamMultiFunction IModelFunction;
   typedef  ROOT::Math::IParamMultiGradFunction IGradModelFunction;

   /** Chi2 Functions */

   /** 
       evaluate the Chi2 given a model function and the data at the point x. 
       return also nPoints as the effective number of used points in the Chi2 evaluation
   */ 
   double EvaluateChi2(const IModelFunction & func, const BinData & data, const double * x, unsigned int & nPoints);  

   /** 
       evaluate the effective Chi2 given a model function and the data at the point x. 
       The effective chi2 uses the errors on the coordinates : W = 1/(sigma_y**2 + ( sigma_x_i * df/dx_i )**2 )
       return also nPoints as the effective number of used points in the Chi2 evaluation
   */ 
   double EvaluateChi2Effective(const IModelFunction & func, const BinData & data, const double * x, unsigned int & nPoints);  

   /** 
       evaluate the Chi2 gradient given a model function and the data at the point x. 
       return also nPoints as the effective number of used points in the Chi2 evaluation
   */ 
   void EvaluateChi2Gradient(const IModelFunction & func, const BinData & data, const double * x, double * grad, unsigned int & nPoints);  

   /** 
       evaluate the LogL given a model function and the data at the point x. 
       return also nPoints as the effective number of used points in the LogL evaluation
   */ 
   double EvaluateLogL(const IModelFunction & func, const UnBinData & data, const double * x, int iWeight, bool extended, unsigned int & nPoints);  

   /** 
       evaluate the LogL gradient given a model function and the data at the point x. 
       return also nPoints as the effective number of used points in the LogL evaluation
   */ 
   void EvaluateLogLGradient(const IModelFunction & func, const UnBinData & data, const double * x, double * grad, unsigned int & nPoints);  

   /** 
       evaluate the Poisson LogL given a model function and the data at the point x. 
       return also nPoints as the effective number of used points in the LogL evaluation
       By default is extended, pass extedend to false if want to be not extended (MultiNomial)
   */ 
   double EvaluatePoissonLogL(const IModelFunction & func, const BinData & data, const double * x, int iWeight, bool extended, unsigned int & nPoints);  

   /** 
       evaluate the Poisson LogL given a model function and the data at the point x. 
       return also nPoints as the effective number of used points in the LogL evaluation
   */ 
   void EvaluatePoissonLogLGradient(const IModelFunction & func, const BinData & data, const double * x, double * grad);  

//    /** 
//        Parallel evaluate the Chi2 given a model function and the data at the point x. 
//        return also nPoints as the effective number of used points in the Chi2 evaluation
//    */ 
//    double ParallelEvalChi2(const IModelFunction & func, const BinData & data, const double * x, unsigned int & nPoints);  

   // methods required by dedicate minimizer like Fumili 
 
   /** 
       evaluate the residual contribution to the Chi2 given a model function and the BinPoint data 
       and if the pointer g is not null evaluate also the gradient of the residual.
       If the function provides parameter derivatives they are used otherwise a simple derivative calculation 
       is used       
   */ 
   double EvaluateChi2Residual(const IModelFunction & func, const BinData & data, const double * x, unsigned int ipoint, double *g = 0);  

   /** 
       evaluate the pdf contribution to the LogL given a model function and the BinPoint data.
       If the pointer g is not null evaluate also the gradient of the pdf.
       If the function provides parameter derivatives they are used otherwise a simple derivative calculation 
       is used 
   */ 
   double EvaluatePdf(const IModelFunction & func, const UnBinData & data, const double * x, unsigned int ipoint, double * g = 0); 

   /** 
       evaluate the pdf contribution to the Poisson LogL given a model function and the BinPoint data. 
       If the pointer g is not null evaluate also the gradient of the Poisson pdf.
       If the function provides parameter derivatives they are used otherwise a simple derivative calculation 
       is used 
   */ 
   double EvaluatePoissonBinPdf(const IModelFunction & func, const BinData & data, const double * x, unsigned int ipoint, double * g = 0);  


   




} // end namespace FitUtil 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_FitUtil */
