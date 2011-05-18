// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_FunctionMember
#define Reflex_FunctionMember

// Include files
#include "Reflex/internal/MemberBase.h"


namespace Reflex {
// forward declarations
class Type;
class Object;
class DictionaryGenerator;


/**
 * @class FunctionMember FunctionMember.h Reflex/FunctionMember.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class FunctionMember: public MemberBase {
public:
   /** default constructor */
   FunctionMember(const char* nam,
                  const Type& typ,
                  StubFunction stubFP,
                  void* stubCtx = 0,
                  const char* params = 0,
                  unsigned int modifiers = 0,
                  TYPE memType = FUNCTIONMEMBER);


   /** destructor */
   virtual ~FunctionMember() {}


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   virtual void GenerateDict(DictionaryGenerator& generator) const;


   /** return full Name of function MemberAt */
   std::string Name(unsigned int mod = 0) const;


   /** Invoke the function (if return At as void*) */

   /*Object Invoke( const Object & obj,
      const std::vector < Object > & paramList ) const;*/
   void Invoke(const Object& obj,
               Object* ret,
               const std::vector<void*>& paramList =
                  std::vector<void*>()) const;


   /** Invoke the function (for static functions) */
   //Object Invoke( const std::vector < Object > & paramList ) const;
   void Invoke(Object* ret,
               const std::vector<void*>& paramList =
                  std::vector<void*>()) const;


   /** number of parameters */
   size_t FunctionParameterSize(bool required = false) const;


   /** FunctionParameterAt nth default value if declared*/
   std::string FunctionParameterDefaultAt(size_t nth) const;


   virtual StdString_Iterator FunctionParameterDefault_Begin() const;
   virtual StdString_Iterator FunctionParameterDefault_End() const;
   virtual Reverse_StdString_Iterator FunctionParameterDefault_RBegin() const;
   virtual Reverse_StdString_Iterator FunctionParameterDefault_REnd() const;


   /** FunctionParameterAt nth Name if declared*/
   std::string FunctionParameterNameAt(size_t nth) const;


   virtual StdString_Iterator FunctionParameterName_Begin() const;
   virtual StdString_Iterator FunctionParameterName_End() const;
   virtual Reverse_StdString_Iterator FunctionParameterName_RBegin() const;
   virtual Reverse_StdString_Iterator FunctionParameterName_REnd() const;


   /** return a pointer to the context */
   void* Stubcontext() const;


   /** return the pointer to the stub function */
   StubFunction Stubfunction() const;


   /**
    * UpdateFunctionParameterNames updates the names of parameters
    * @param  parameters new list of ';' separated parameter names, must not specify default values
    */
   virtual void UpdateFunctionParameterNames(const char* parameters);

private:
   /** pointer to the stub function */
   StubFunction fStubFP;


   /** user data for the stub function */
   void* fStubCtx;


   /** FunctionParameterAt names */
   mutable
   std::vector<std::string> fParameterNames;


   /** FunctionParameterAt names */
   mutable
   std::vector<std::string> fParameterDefaults;


   /** number of required parameters */
   size_t fReqParameters;

};    // class FunctionMember
} //namespace Reflex

//-------------------------------------------------------------------------------
inline std::string
Reflex::FunctionMember::FunctionParameterDefaultAt(size_t nth) const {
//-------------------------------------------------------------------------------
   return fParameterDefaults[nth];
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::FunctionMember::FunctionParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::FunctionMember::FunctionParameterDefault_End() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::FunctionMember::FunctionParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string> &)fParameterDefaults).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::FunctionMember::FunctionParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string> &)fParameterDefaults).rend();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::FunctionMember::FunctionParameterName_Begin() const {
//-------------------------------------------------------------------------------
   return fParameterNames.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::FunctionMember::FunctionParameterName_End() const {
//-------------------------------------------------------------------------------
   return fParameterNames.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::FunctionMember::FunctionParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string> &)fParameterNames).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::FunctionMember::FunctionParameterName_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string> &)fParameterNames).rend();
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::FunctionMember::FunctionParameterNameAt(size_t nth) const {
//-------------------------------------------------------------------------------
   return fParameterNames[nth];
}


//-------------------------------------------------------------------------------
inline void*
Reflex::FunctionMember::Stubcontext() const {
//-------------------------------------------------------------------------------
   return fStubCtx;
}


//-------------------------------------------------------------------------------
inline Reflex::StubFunction
Reflex::FunctionMember::Stubfunction() const {
//-------------------------------------------------------------------------------
   return fStubFP;
}


#endif // Reflex_FunctionMember
