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

#ifndef ROOT_Math_FitMethodFunction
#include "Math/FitMethodFunction.h"
#endif

#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif


#include <memory>



namespace ROOT {


   namespace Fit {



//___________________________________________________________________________________
/**
   BasicFCN class: base class  for the objective functions used in the fits 
   It has a reference to the data and th emodel function used in the fit. 
   It cannot be instantiated but constructed from the derived classes 
*/
template<class FunType, class DataType>
class BasicFCN : public ::ROOT::Math::BasicFitMethodFunction<FunType> {

protected:

   
   typedef  ::ROOT::Math::BasicFitMethodFunction<FunType> BaseObjFunction;
   typedef typename  BaseObjFunction::BaseFunction BaseFunction;

   typedef  ::ROOT::Math::IParamMultiFunction IModelFunction;

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
