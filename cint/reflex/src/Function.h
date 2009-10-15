// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Function
#define Reflex_Function

// Include files
#include "Reflex/internal/TypeBase.h"
#include "Reflex/Type.h"

namespace Reflex {
// forward declarations


/**
 * @class Function Function.h Reflex/Function.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class Function: public TypeBase {
public:
   /** default constructor */
   Function(const Type& retType,
            const std::vector<Type>& parameters,
            const std::type_info& ti,
            TYPE functionType = FUNCTION);


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
   virtual std::string Name(unsigned int mod = 0) const;


   /**
    * FunctionParameterAt returns the nth FunctionParameterAt
    * @param  nth nth FunctionParameterAt
    * @return pointer to nth FunctionParameterAt At
    */
   virtual Type FunctionParameterAt(size_t nth) const;


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
   virtual Type ReturnType() const;


   /** static function that composes the At Name */
   static std::string BuildTypeName(const Type& ret,
                                    const std::vector<Type>& param,
                                    unsigned int mod = SCOPED | QUALIFIED);

private:
   /**
    * container of parameter types
    * @label function parameter types
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 0..*
    */
   mutable
   std::vector<Type> fParameters;


   /**
    * return type
    * @label return type
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   Type fReturnType;


   /** modifiers of function and return At */
   unsigned int fModifiers;

};    // class Function
} // namespace Reflex


//-------------------------------------------------------------------------------
inline bool
Reflex::Function::IsVirtual() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VIRTUAL);
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Function::FunctionParameterSize() const {
//-------------------------------------------------------------------------------
   return fParameters.size();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Function::FunctionParameter_Begin() const {
//-------------------------------------------------------------------------------
   return fParameters.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Function::FunctionParameter_End() const {
//-------------------------------------------------------------------------------
   return fParameters.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Function::FunctionParameter_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Type> &)fParameters).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Function::FunctionParameter_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Type> &)fParameters).rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Function::FunctionParameterAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (nth < fParameters.size()) {
      return fParameters[nth];
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Function::ReturnType() const {
//-------------------------------------------------------------------------------
   return fReturnType;
}


#endif // Reflex_Function
