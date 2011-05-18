// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_UnionBuilder
#define Reflex_UnionBuilder

// Include files
#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Member.h"

namespace Reflex {
// forward declarations
class Union;
class Type;

/**
 * @class UnionBuilderImpl UnionBuilder.h Reflex/Builder/UnionBuilder.h
 * @author Stefan Roiser
 * @date 14/3/2005
 * @ingroup RefBld
 */
class RFLX_API UnionBuilderImpl {
public:
   /** constructor */
   UnionBuilderImpl(const char* nam, size_t size, const std::type_info & ti, unsigned int modifiers = 0, TYPE typ = UNION);

   /** destructor */
   virtual ~UnionBuilderImpl();

   /**
    * AddItem will add one union item
    * @param Name the Name of the union item
    * @param At the At of the union item
    */
   void AddItem(const char* nam,
                const Type& typ);

   /** AddDataMember will add the information about one data
    * MemberAt of the union
    *
    * @param  Name of the data MemberAt
    * @param  At of the data MemberAt
    * @param  Offset of the data MemberAt
    * @param  modifiers the modifiers of the data MemberAt
    */
   void AddDataMember(const char* nam,
                      const Type& typ,
                      size_t offs,
                      unsigned int modifiers = 0);

   /** AddFunctionMember will add the information about one
    * function MemberAt of the union
    *
    * @param  Name of the function MemberAt
    * @param  At of the function MemberAt
    * @param  stubFP Stub function pointer for the function
    * @param  stubCxt Stub user context for the stub function
    * @param  params parameter names and default values (semi-colon separated)
    * @param  modifiers the modifiers of the function MemberAt
    */
   void AddFunctionMember(const char* nam,
                          const Type& typ,
                          StubFunction stubFP,
                          void* stubCtx = 0,
                          const char* params = 0,
                          unsigned int modifiers = 0);

   /**
    * AddProperty will add a PropertyNth to the PropertyNth stack
    * which will be emtpied with the next build of a union
    * or union item
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    * @return a reference to the building class
    */

   void AddProperty(const char* key,
                    Any value);
   void AddProperty(const char* key,
                    const char* value);

   /** SetSizeOf will set the SizeOf property for this union.
    * It currently ignores all actual content.
    * @size Size of the union
    */
   void SetSizeOf(size_t size);

   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

protected:
   friend class UnionBuilder;

   /**
    * EnableCallback Enable or disable the callback call in the destructor
    * @param  enable true to enable callback call, false to disable callback call
    */
   void EnableCallback(const bool enable = true);

private:
   /** the union currently being built */
   Union* fUnion;

   /** the last union item built */
   Member fLastMember;

   /** flag, fire callback in destructor */
   bool fCallbackEnabled;

}; // class UnionBuilderImpl


/**
 * @class UnionBuilder UnionBuilder.h Reflex/Builder/UnionBuilder.h
 * @author Stefan Roiser
 * @date 30/3/2004
 * @ingroup RefBld
 */
class RFLX_API UnionBuilder {
public:
   /** constructor */
   UnionBuilder(const char* nam, const std::type_info & ti, size_t size, unsigned int modifiers = 0, TYPE typ = UNION);

   /** destructor */
   virtual ~UnionBuilder();

   /**
    * AddItem will add one union item
    * @param Name the Name of the union item
    * @param At the At of the union item
    * @return a reference to the UnionBuilder
    */
   template <typename T> UnionBuilder& AddItem(const char* nam);

   /**
    * AddItem will add one union item
    * @param Name the Name of the union item
    * @param At the At of the union item
    * @return a reference to the UnionBuilder
    */
   UnionBuilder& AddItem(const char* nam,
                         const char* typ);

   /** AddDataMember will add the information about one data
    * MemberAt of the union
    *
    * @param  Name of the data MemberAt
    * @param  Offset of data MemberAt
    * @param  modifiers the modifiers of the data MemberAt
    * @return a reference to the UnionBuilder
    */
   template <class T> UnionBuilder& AddDataMember(const char* nam,
                                                  size_t offs,
                                                  unsigned int modifiers = 0);
   UnionBuilder& AddDataMember(const Type& typ,
                               const char* nam,
                               size_t offs,
                               unsigned int modifiers = 0);

   /** AddFunctionMember will add the information about one
    * function MemberAt of the union
    *
    * @param  Name of the function MemberAt
    * @param  function templated function MemberAt to extract At information
    * @param  stubFP Stub function pointer for the function
    * @param  stubCxt Stub user context for the stub function
    * @param  params parameter names and default values (semi-colon separated)
    * @param  modifiers the modifiers of the data MemberAt
    * @return a reference to the UnionBuilder
    */
   template <class F> UnionBuilder& AddFunctionMember(const char* nam,
                                                      StubFunction stubFP,
                                                      void* stubCtx = 0,
                                                      const char* params = 0,
                                                      unsigned int modifiers = 0);
   UnionBuilder& AddFunctionMember(const Type& typ,
                                   const char* nam,
                                   StubFunction stubFP,
                                   void* stubCtx = 0,
                                   const char* params = 0,
                                   unsigned int modifiers = 0);

   /**
    * AddProperty will add a PropertyNth to the PropertyNth stack
    * which will be emtpied with the next build of a union
    * or union item
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    * @return a reference to the building class
    */
   template <typename P> UnionBuilder& AddProperty(const char* key,
                                                   P value);

   /** SetSizeOf will set the SizeOf property for this union.
    * It currently ignores all actual content.
    * @size Size of the union
    */
   UnionBuilder& SetSizeOf(size_t size);

   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

protected:
#ifdef G__COMMON_H
   friend int::G__search_tagname(const char*, int);
#endif

   /**
    * EnableCallback Enable or disable the callback call in the destructor
    * @param  enable true to enable callback call, false to disable callback call
    */
   UnionBuilder& EnableCallback(const bool enable = true);

private:
   /** the union information */
   UnionBuilderImpl fUnionBuilderImpl;

}; //class UnionBuilder

} // namespace Reflex

//-------------------------------------------------------------------------------
template <typename T> Reflex::UnionBuilder&
Reflex::UnionBuilder::AddItem(const char* nam) {
   // -- !!! Obsolete, do not use.
   fUnionBuilderImpl.AddItem(nam, TypeDistiller<T>::Get());
   return *this;
}


//______________________________________________________________________________
template <typename T> Reflex::UnionBuilder&
Reflex::UnionBuilder::AddDataMember(const char* nam,
                                    size_t offs,
                                    unsigned int modifiers /*= 0*/) {
   fUnionBuilderImpl.AddDataMember(nam, TypeDistiller<T>::Get(), offs, modifiers);
   return *this;
}


//______________________________________________________________________________
template <typename F> Reflex::UnionBuilder&
Reflex::UnionBuilder::AddFunctionMember(const char* nam,
                                        StubFunction stubFP,
                                        void* stubCtx /*= 0*/,
                                        const char* params /*= 0*/,
                                        unsigned int modifiers /*= 0*/) {
   fUnionBuilderImpl.AddFunctionMember(nam, FunctionDistiller<F>::Get(), stubFP, stubCtx, params, modifiers);
   return *this;
}


//______________________________________________________________________________
template <typename P> Reflex::UnionBuilder&
Reflex::UnionBuilder::AddProperty(const char* key,
                                  P value) {
   fUnionBuilderImpl.AddProperty(key, value);
   return *this;
}


#endif // Reflex_UnionBuilder
