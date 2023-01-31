// @(#)root/mathcore:$Id$
// Author: L. Moneta 25 Nov 2014

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class BasicFCN

#ifndef ROOT_Fit_BasicFCN
#define ROOT_Fit_BasicFCN

#include "Math/FitMethodFunction.h"

#include "Math/IParamFunction.h"

#include "Math/IParamFunctionfwd.h"

#include <memory>



namespace ROOT {


   namespace Fit {



//___________________________________________________________________________________
/**
   BasicFCN class: base class  for the objective functions used in the fits
   It has a reference to the data and the model function used in the fit.
   It cannot be instantiated but constructed from the derived classes
*/
template<class DerivFunType, class ModelFunType, class DataType>
class BasicFCN : public ::ROOT::Math::BasicFitMethodFunction<DerivFunType> {

protected:

   typedef typename ModelFunType::BackendType T;

   typedef  ::ROOT::Math::BasicFitMethodFunction<DerivFunType> BaseObjFunction;
   typedef typename  BaseObjFunction::BaseFunction BaseFunction;

   typedef  ::ROOT::Math::IParamMultiFunctionTempl<T> IModelFunction;
   typedef  ::ROOT::Math::IParametricGradFunctionMultiDimTempl<T> IGradModelFunction;

   /**
      Constructor from data set  and model function
   */
   BasicFCN (const std::shared_ptr<DataType> & data, const std::shared_ptr<IModelFunction> & func) :
      BaseObjFunction(func->NPar(), data->Size() ),
      fData(data),
      fFunc(func)
   { }



   /**
      Destructor (no operations)
   */
   virtual ~BasicFCN ()  {}

public:


   /// access to const reference to the data
   virtual const DataType & Data() const { return *fData; }

   /// access to data pointer
   std::shared_ptr<DataType> DataPtr() const { return fData; }

   /// access to const reference to the model function
   virtual const IModelFunction & ModelFunction() const { return *fFunc; }

   /// access to function pointer
   std::shared_ptr<IModelFunction> ModelFunctionPtr() const { return fFunc; }

   /// flag to indicate if can compute Hessian
   virtual bool HasHessian() const {
      if (!BaseObjFunction::IsAGradFCN()) return false;
      auto gfunc = dynamic_cast<const IGradModelFunction *>( fFunc.get());
      if (!gfunc) return false;
      return gfunc->HasParameterHessian();
   }




protected:


   /// Set the data pointer
   void SetData(const std::shared_ptr<DataType> & data) { fData = data; }

      /// Set the function pointer
   void SetModelFunction(const std::shared_ptr<IModelFunction> & func) { fFunc = func; }


   std::shared_ptr<DataType>  fData;
   std::shared_ptr<IModelFunction>  fFunc;



};



   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_BasicFCN */
