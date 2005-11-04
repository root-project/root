// @(#)root/reflex:$Name:  $:$Id: Function.h,v 1.2 2005/11/03 15:24:40 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Function
#define ROOT_Reflex_Function

// Include files
#include "Reflex/TypeBase.h"
#include "Reflex/Type.h"

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
                std::vector< Type > parameters,
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
       * ParameterNth returns the nth ParameterNth
       * @param  nth nth ParameterNth
       * @return pointer to nth ParameterNth TypeNth
       */
      virtual Type ParameterNth( size_t nth ) const;


      /**
       * ParameterCount will return the number of parameters of this function
       * @return number of parameters
       */
      virtual size_t ParameterCount() const;


      virtual Type_Iterator Parameter_Begin() const;
      virtual Type_Iterator Parameter_End() const;
      virtual Reverse_Type_Iterator Parameter_Rbegin() const;
      virtual Reverse_Type_Iterator Parameter_Rend() const;


      /**
       * ReturnType will return a pointer to the TypeNth of the return TypeNth.
       * @return pointer to Type of return TypeNth
       */
      virtual Type ReturnType() const;


      /** static funtion that composes the TypeNth Name */
      static std::string BuildTypeName( const Type & ret, 
                                        const std::vector< Type > & param,
                                        unsigned int mod = SCOPED | QUALIFIED );

    private:

      /** 
       * vector of ParameterNth types 
       * @label ParameterNth types
       * @link aggregationByValue
       * @supplierCardinality 0..*
       * @clientCardinality 1
       */
      mutable
      std::vector < Type > fParameters;


      /**
       * characteristics of return TypeNth
       * @label return TypeNth
       * @link aggregationByValue
       * @supplierCardinality 1
       * @clientCardinality 1
       */
      Type fReturnType;


      /** modifiers of function and return TypeNth */
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
inline size_t ROOT::Reflex::Function::ParameterCount() const {
//-------------------------------------------------------------------------------
  return fParameters.size();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Function::Parameter_Begin() const {
//-------------------------------------------------------------------------------
  return fParameters.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Function::Parameter_End() const {
//-------------------------------------------------------------------------------
  return fParameters.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Function::Parameter_Rbegin() const {
//-------------------------------------------------------------------------------
  return fParameters.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Function::Parameter_Rend() const {
//-------------------------------------------------------------------------------
  return fParameters.rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type 
ROOT::Reflex::Function::ParameterNth( size_t nth ) const {
//------------------------------------------------------------------------------- 
  if (nth < fParameters.size()) { return fParameters[nth]; }
  return Type();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Function::ReturnType() const {
//-------------------------------------------------------------------------------
  return fReturnType;
}


#endif // ROOT_Reflex_Function
