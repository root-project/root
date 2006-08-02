// @(#)root/reflex:$Name:  $:$Id: Function.h,v 1.8 2006/08/01 09:14:33 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Function
#define ROOT_Reflex_Function

// Include files
#include "Reflex/internal/TypeBase.h"
#include "Reflex/internal/OwnedType.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations


      /** 
       * @class Function Function.h Reflex/Function.h
       * @author Stefan Roiser
       * @date 24/11/2003 
       * @ingroup Ref
       */
      class Function : public TypeBase {

      public:


         /** default constructor */
         Function( const Type & retType,
                   const std::vector< Type > & parameters,
                   const std::type_info & ti,
                   TYPE functionType = FUNCTION );


         /** destructor */
         virtual ~Function() {}


         /**
          * IsVirtual will return true if the class contains a virtual table
          * @return true if the class contains a virtual table
          */
         virtual bool IsVirtual() const;


         /**
          * Name will return the Name of the function
          * @param  mod modifiers to be applied when generating the Name
          * @return Name of function
          */
         virtual std::string Name( unsigned int mod = 0 ) const;


         /**
          * FunctionParameterAt returns the nth FunctionParameterAt
          * @param  nth nth FunctionParameterAt
          * @return pointer to nth FunctionParameterAt At
          */
         virtual const Type & FunctionParameterAt( size_t nth ) const;


         /**
          * FunctionParameterSize will return the number of parameters of this function
          * @return number of parameters
          */
         virtual size_t FunctionParameterSize() const;


         virtual Type_Iterator FunctionParameter_Begin() const;
         virtual Type_Iterator FunctionParameter_End() const;
         virtual Reverse_Type_Iterator FunctionParameter_RBegin() const;
         virtual Reverse_Type_Iterator FunctionParameter_REnd() const;


         /**
          * ReturnType will return a pointer to the At of the return At.
          * @return pointer to Type of return At
          */
         virtual const Type & ReturnType() const;


         /** static funtion that composes the At Name */
         static std::string BuildTypeName( const Type & ret, 
                                           const std::vector< Type > & param,
                                           unsigned int mod = SCOPED | QUALIFIED );

      private:

         /** 
          * vector of FunctionParameterAt types 
          * @label FunctionParameterAt types
          * @link aggregationByValue
          * @supplierCardinality 0..*
          * @clientCardinality 1
          */
         mutable
            std::vector < Type > fParameters;


         /**
          * characteristics of return At
          * @label return At
          * @link aggregationByValue
          * @supplierCardinality 1
          * @clientCardinality 1
          */
         Type fReturnType;


         /** modifiers of function and return At */
         unsigned int fModifiers;

      }; // class Function
   } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Function::IsVirtual() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VIRTUAL);
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Function::FunctionParameterSize() const {
//-------------------------------------------------------------------------------
   return fParameters.size();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Function::FunctionParameter_Begin() const {
//-------------------------------------------------------------------------------
   return fParameters.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Function::FunctionParameter_End() const {
//-------------------------------------------------------------------------------
   return fParameters.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Function::FunctionParameter_RBegin() const {
//-------------------------------------------------------------------------------
   return fParameters.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Function::FunctionParameter_REnd() const {
//-------------------------------------------------------------------------------
   return fParameters.rend();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & 
ROOT::Reflex::Function::FunctionParameterAt( size_t nth ) const {
//------------------------------------------------------------------------------- 
   if (nth < fParameters.size()) { return fParameters[nth]; }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Function::ReturnType() const {
//-------------------------------------------------------------------------------
   return fReturnType;
}


#endif // ROOT_Reflex_Function
