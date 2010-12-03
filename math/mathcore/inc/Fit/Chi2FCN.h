// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Sep  5 09:13:32 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class Chi2FCN

#ifndef ROOT_Fit_Chi2FCN
#define ROOT_Fit_Chi2FCN

#ifndef ROOT_Math_FitMethodFunction
#include "Math/FitMethodFunction.h"
#endif

#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif


#ifndef ROOT_Fit_BinData
#include "Fit/BinData.h"
#endif


#ifndef ROOT_Fit_FitUtil
#include "Fit/FitUtil.h"
#endif

//#define ROOT_FIT_PARALLEL

#ifdef ROOT_FIT_PARALLEL
#ifndef ROOT_Fit_FitUtilParallel
#include "Fit/FitUtilParallel.h"
#endif
#endif

/** 
@defgroup FitMethodFunc Fit Method Classes 

Classes describing Fit Method functions
@ingroup Fit
*/


namespace ROOT { 


   namespace Fit { 


template<class FunType> 
struct ModelFunctionTrait { 
   typedef  ::ROOT::Math::IParamMultiFunction ModelFunc;
};
template<>      
struct ModelFunctionTrait<ROOT::Math::IMultiGradFunction>  { 
   typedef  ::ROOT::Math::IParamMultiGradFunction ModelFunc;
};



//___________________________________________________________________________________
/** 
   Chi2FCN class for binnned fits using the least square methods 

   @ingroup  FitMethodFunc   
*/ 
template<class FunType> 
class Chi2FCN : public ::ROOT::Math::BasicFitMethodFunction<FunType> {

public: 



   typedef  ::ROOT::Math::BasicFitMethodFunction<FunType> BaseObjFunction; 
   typedef typename  BaseObjFunction::BaseFunction BaseFunction; 

   typedef  typename ModelFunctionTrait<FunType>::ModelFunc IModelFunction;
   typedef typename BaseObjFunction::Type_t Type_t;

   /** 
      Constructor from data set (binned ) and model function 
   */ 
   Chi2FCN (const BinData & data, const IModelFunction & func) : 
      BaseObjFunction(func.NPar(), data.Size() ),
      fData(data), 
      fFunc(func), 
      fNEffPoints(0),
      fGrad ( std::vector<double> ( func.NPar() ) )
   { }

   /** 
      Destructor (no operations)
   */ 
   virtual ~Chi2FCN ()  {}

#ifdef LATER
private:

   // usually copying is non trivial, so we make this unaccessible

   /** 
      Copy constructor
   */ 
   Chi2FCN(const Chi2FCN &); 

   /** 
      Assignment operator
   */ 
   Chi2FCN & operator = (const Chi2FCN & rhs); 

#endif
public: 

   virtual BaseFunction * Clone() const { 
      // clone the function
      Chi2FCN * fcn =  new Chi2FCN(fData,fFunc); 
      return fcn; 
   }
 


   using BaseObjFunction::operator();


   // effective points used in the fit (exclude the rejected one)
   virtual unsigned int NFitPoints() const { return fNEffPoints; }


   /// i-th chi-square residual  
   virtual double DataElement(const double * x, unsigned int i, double * g) const { 
      if (i==0) this->UpdateNCalls();
      return FitUtil::EvaluateChi2Residual(fFunc, fData, x, i, g); 
   }

   // need to be virtual to be instantited
   virtual void Gradient(const double *x, double *g) const { 
      // evaluate the chi2 gradient
      FitUtil::EvaluateChi2Gradient(fFunc, fData, x, g, fNEffPoints);
   }

   /// get type of fit method function
   virtual  typename BaseObjFunction::Type_t Type() const { return BaseObjFunction::kLeastSquare; }

   /// access to const reference to the data 
   virtual const BinData & Data() const { return fData; }

   /// access to const reference to the model function
   virtual const IModelFunction & ModelFunction() const { return fFunc; }



protected: 


   /// set number of fit points (need to be called in const methods, make it const) 
   virtual void SetNFitPoints(unsigned int n) const { fNEffPoints = n; }

private: 

   /**
      Evaluation of the  function (required by interface)
    */
   virtual double DoEval (const double * x) const { 
      this->UpdateNCalls();
#ifdef ROOT_FIT_PARALLEL
      return FitUtilParallel::EvaluateChi2(fFunc, fData, x, fNEffPoints); 
#else 
      if (!fData.HaveCoordErrors() ) 
         return FitUtil::EvaluateChi2(fFunc, fData, x, fNEffPoints); 
      else 
         return FitUtil::EvaluateChi2Effective(fFunc, fData, x, fNEffPoints); 
#endif
   } 

   // for derivatives 
   virtual double  DoDerivative(const double * x, unsigned int icoord ) const { 
      Gradient(x, &fGrad[0]); 
      return fGrad[icoord]; 
   }

   const BinData & fData; 
   const IModelFunction & fFunc; 

   mutable unsigned int fNEffPoints;  // number of effective points used in the fit 

   mutable std::vector<double> fGrad; // for derivatives


}; 

      // define useful typedef's
      typedef Chi2FCN<ROOT::Math::IMultiGenFunction> Chi2Function; 
      typedef Chi2FCN<ROOT::Math::IMultiGradFunction> Chi2GradFunction; 


   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_Chi2FCN */
