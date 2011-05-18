// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_ClassBuilder
#define Reflex_ClassBuilder

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Tools.h"
#include "Reflex/Member.h"
#include "Reflex/Callback.h"
#include "Reflex/Builder/TypeBuilder.h"

// Forward declaration for 'friendship' purpose.
namespace Cint { namespace Internal {} }

namespace Reflex {

/* Helper class to avoid compiler warning about casting function pointer
** to void pointer.
*/
class BuilderFunc2Void {
   typedef void (*funcptr_t)();

   union funcptr_and_voidptr {
      typedef void (*funcptr_t)();

      funcptr_and_voidptr(void *val) : _read(val) {}

      void *_read;
      funcptr_t _write;
   };

   funcptr_and_voidptr _tmp;
public:
   template <typename T>
   BuilderFunc2Void( T vfp ) : _tmp(0) {
      _tmp._write = ( funcptr_t )vfp;
   }

   operator void*() const {
      return _tmp._read;
   }
};

// forward declarations
class Class;
class ClassBuilder;
class OnDemandBuilderForScope;

template <typename C> class ClassBuilderT;

/**
 * @class ClassBuilderImpl ClassBuilder.h Reflex/Builder/ClassBuilder.h
 * @author Stefan Roiser
 * @date 30/3/2004
 * @ingroup RefBld
 */
class RFLX_API ClassBuilderImpl {
public:
   /** constructor */
   ClassBuilderImpl(const char* nam, const std::type_info & ti, size_t size, unsigned int modifiers = 0, TYPE typ = CLASS);
   ClassBuilderImpl(Class* cl);

   /** destructor */
   virtual ~ClassBuilderImpl();

   /**
    * AddBase will add the information about one BaseAt class
    * @param  Name of the BaseAt class
    * @param  OffsetFP function pointer for Offset calculation
    * @param  modifiers the modifiers of the class
    */
   void AddBase(const Type& bas,
                OffsetFunction offsFP,
                unsigned int modifiers = 0);

   /** AddDataMember will add the information about one data
    * MemberAt of the class
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
    * function MemberAt of the class
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

   void AddTypedef(const Type& typ,
                   const char* def);

   void AddEnum(const char* nam,
                const char* values,
                const std::type_info* ti,
                unsigned int modifiers = 0);

   // This is for anonymous union support.
   //void addUnion(const char* nam, const char* values, const std::type_info& ti, unsigned int modifiers = 0);


   /** AddProperty will add a PropertyNth to the PropertyNth stack
    * which will be emtpied with the next call of a builder
    * class and attached to the item built with this call
    *
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    */
   void AddProperty(const char* key,
                    Any value);
   void AddProperty(const char* key,
                    const char* value);
   void AddProperty(const char* key,
                    const Literal& value) { AddProperty(key, (const char*)value); }


   /**
    * Register an on demand builder for data members with this class.
    */
   void AddOnDemandDataMemberBuilder(OnDemandBuilderForScope* odb);

   /**
    * Register an on demand builder for function members with this class.
    */
   void AddOnDemandFunctionMemberBuilder(OnDemandBuilderForScope* odb);

   /** SetSizeOf will set the SizeOf property for this class.
    * It currently ignore all actual content.
    * @size Size of the class
    */
   void SetSizeOf(size_t size);

   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

protected:
   friend class ClassBuilder;
   template <class C> friend class ClassBuilderT;

   /**
    * EnableCallback Enable or disable the callback call in the destructor
    * @param  enable true to enable callback call, false to disable callback call
    */
   void EnableCallback(bool enable = true);

private:
   /** current class being built */
   Class* fClass;

   /** last added MemberAt */
   Member fLastMember;

   /** flag, true if this is truly building a new class */
   bool fNewClass;

   /** flag, fire callback in destructor */
   bool fCallbackEnabled;

};    // class ClassBuilderImpl


/**
 * @class ClassBuilder ClassBuilder.h Reflex/Builder/ClassBuilder.h
 * @author Stefan Roiser
 * @date 24/5/2004
 * @ingroup RefBld
 */
class RFLX_API ClassBuilder {
public:
   /** constructor */
   ClassBuilder(const char* nam, const std::type_info & ti, size_t size, unsigned int modifiers = 0, TYPE typ = CLASS);
   ClassBuilder(Class* cl);

   /** destructor */
   virtual ~ClassBuilder();

   /**
    * AddBase will add the information about one BaseAt class
    * @param  Name of the BaseAt class
    * @param  OffsetFP function pointer for Offset calculation
    * @param  modifiers the modifiers of the class
    */
   template <class C, class B> ClassBuilder& AddBase(unsigned int modifiers = 0);
   ClassBuilder& AddBase(const Type& bas,
                         OffsetFunction offsFP,
                         unsigned int modifiers = 0);

   /** AddDataMember will add the information about one data
    * MemberAt of the class
    *
    * @param  Name of the data MemberAt
    * @param  Offset of data MemberAt
    * @param  modifiers the modifiers of the data MemberAt
    * @return a reference to the ClassBuilder
    */
   template <class T> ClassBuilder& AddDataMember(const char* nam,
                                                  size_t offs,
                                                  unsigned int modifiers = 0);
   ClassBuilder& AddDataMember(const Type& typ,
                               const char* nam,
                               size_t offs,
                               unsigned int modifiers = 0);

   /** AddFunctionMember will add the information about one
    * function MemberAt of the class
    *
    * @param  Name of the function MemberAt
    * @param  function templated function MemberAt to extract At information
    * @param  stubFP Stub function pointer for the function
    * @param  stubCxt Stub user context for the stub function
    * @param  params pamater names and default values (semi-colon separated)
    * @param  modifiers the modifiers of the data MemberAt
    * @return a reference to the ClassBuilder
    */
   template <class F> ClassBuilder& AddFunctionMember(const char* nam,
                                                      StubFunction stubFP,
                                                      void* stubCtx = 0,
                                                      const char* params = 0,
                                                      unsigned int modifiers = 0);
   ClassBuilder& AddFunctionMember(const Type& typ,
                                   const char* nam,
                                   StubFunction stubFP,
                                   void* stubCtx = 0,
                                   const char* params = 0,
                                   unsigned int modifiers = 0);

   template <typename TD> ClassBuilder& AddTypedef(const char* def);
   ClassBuilder& AddTypedef(const Type& typ,
                            const char* def);
   ClassBuilder& AddTypedef(const char* typ,
                            const char* def);

   template <typename E> ClassBuilder& AddEnum(const char* values,
                                               unsigned int modifiers = 0);

   ClassBuilder& AddEnum(const char* nam,
                         const char* values,
                         const std::type_info* ti = 0,
                         unsigned int modifiers = 0);

   // This is for anonymous union support.
   //ClassBuilder& addUnion(const char* nam, const char* values, unsigned int modifiers);


   /** AddProperty will add a PropertyNth to the last defined
    * data MemberAt, method or class.
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    * @return a reference to the building class
    */
   template <typename P> ClassBuilder& AddProperty(const char* key,
                                                   P value);

   /**
    * Register an on demand data member builder with this class.
    */
   ClassBuilder& AddOnDemandDataMemberBuilder(OnDemandBuilderForScope* odb);

   /**
    * Register an on demand function member builder with this class.
    */
   ClassBuilder& AddOnDemandFunctionMemberBuilder(OnDemandBuilderForScope* odb);

   /** SetSizeOf will set the SizeOf property for this class.
    * It currently ignore all actual content.
    * @size Size of the class
    */
   ClassBuilder& SetSizeOf(size_t size);

   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

protected:
#ifdef G__COMMON_H
   friend int::G__search_tagname(const char*, int);
   friend void Cint::Internal::G__set_stdio();
   friend void Cint::Internal::G__create_bytecode_arena();
#endif

   /**
    * EnableCallback Enable or disable the callback call in the destructor
    * @param  enable true to enable callback call, false to disable callback call
    */
   ClassBuilder& EnableCallback(bool enable = true);

private:
   ClassBuilderImpl fClassBuilderImpl;

};    // class ClassBuilder


/**
 * @class ClassBuilderT ClassBuilder.h Reflex/Builder/ClassBuilder.h
 * @author Stefan Roiser
 * @date 30/3/2004
 * @ingroup RefBld
 */
template <class C>
class ClassBuilderT {
public:
   /** constructor */
   ClassBuilderT(unsigned int modifiers = 0,
                 TYPE typ = CLASS);


   /** constructor */
   ClassBuilderT(const char* nam,
                 unsigned int modifiers = 0,
                 TYPE typ = CLASS);


   /**
    * AddBase will add the information about one BaseAt class
    * @param  Name of the BaseAt class
    * @param  OffsetFP function pointer for Offset calculation
    * @param  modifiers the modifiers of the class
    */
   template <class B>
   ClassBuilderT& AddBase(unsigned int modifiers = 0);
   ClassBuilderT& AddBase(const Type& bas,
                          OffsetFunction offsFP,
                          unsigned int modifiers = 0);


   /** AddDataMember will add the information about one data
    * MemberAt of the class
    *
    * @param  Name of the data MemberAt
    * @param  Offset of data MemberAt
    * @param  modifiers the modifiers of the data MemberAt
    * @return a reference to the ClassBuilderT
    */
   template <class T>
   ClassBuilderT& AddDataMember(const char* nam,
                                size_t offs,
                                unsigned int modifiers = 0);
   ClassBuilderT& AddDataMember(const Type& typ,
                                const char* nam,
                                size_t offs,
                                unsigned int modifiers = 0);


   /** AddFunctionMember will add the information about one
    * function MemberAt of the class
    *
    * @param  Name of the function MemberAt
    * @param  function templated function MemberAt to extract At information
    * @param  stubFP Stub function pointer for the function
    * @param  stubCxt Stub user context for the stub function
    * @param  params pamater names and default values (semi-colon separated)
    * @param  modifiers the modifiers of the data MemberAt
    * @return a reference to the ClassBuilder
    */
   template <class F>
   ClassBuilderT& AddFunctionMember(const char* nam,
                                    StubFunction stubFP,
                                    void* stubCtx = 0,
                                    const char* params = 0,
                                    unsigned int modifiers = 0);
   ClassBuilderT& AddFunctionMember(const Type& typ,
                                    const char* nam,
                                    StubFunction stubFP,
                                    void* stubCtx = 0,
                                    const char* params = 0,
                                    unsigned int modifiers = 0);

   template <typename TD>
   ClassBuilderT& AddTypedef(const char* def);
   ClassBuilderT& AddTypedef(const Type& typ,
                             const char* def);
   ClassBuilderT& AddTypedef(const char* typ,
                             const char* def);

   template <typename E>
   ClassBuilderT& AddEnum(const char* values,
                          unsigned int modifiers = 0);
   ClassBuilderT& AddEnum(const char* nam,
                          const char* values,
                          const std::type_info* ti = 0,
                          unsigned int modifiers = 0);

   //ClassBuilderT & addUnion( const char * nam,
   //                          const char * values,
   //                          unsigned int modifiers );


   /** AddProperty will add a PropertyNth to the last defined
    * data MemberAt, method or class.
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    * @return a reference to the building class
    */
   template <typename P>
   ClassBuilderT& AddProperty(const char* key,
                              P value);


   /**
    * Register an on demand data member builder with this class.
    */
   void AddOnDemandDataMemberBuilder(OnDemandBuilderForScope* odb);


   /**
    * Register an on demand function member builder with this class.
    */
   void AddOnDemandFunctionMemberBuilder(OnDemandBuilderForScope* odb);


   /** SetSizeOf will set the SizeOf property for this class.
    * It currently ignore all actual content.
    * @size Size of the class
    */
   ClassBuilderT& SetSizeOf(size_t size);

   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

protected:
   /**
    * EnableCallback Enable or disable the callback call in the destructor
    * @param  enable true to enable callback call, false to disable callback call
    */
   ClassBuilderT& EnableCallback(bool enable = true);

private:
   ClassBuilderImpl fClassBuilderImpl;

};    // class ClassBuilderT

} // namespace Reflex

//______________________________________________________________________________
template <typename C, typename B> inline Reflex::ClassBuilder&
Reflex::ClassBuilder::AddBase(unsigned int modifiers) {
   fClassBuilderImpl.AddBase(GetType<B>(), BaseOffset<C, B>::Get(), modifiers);
   return *this;
}


//______________________________________________________________________________
template <typename T> inline Reflex::ClassBuilder&
Reflex::ClassBuilder::AddDataMember(const char* nam,
                                    size_t offs,
                                    unsigned int modifiers) {
   fClassBuilderImpl.AddDataMember(nam, TypeDistiller<T>::Get(), offs, modifiers);
   return *this;
}


//______________________________________________________________________________
template <typename F> inline Reflex::ClassBuilder&
Reflex::ClassBuilder::AddFunctionMember(const char* nam,
                                        StubFunction stubFP,
                                        void* stubCtx,
                                        const char* params,
                                        unsigned int modifiers) {
   fClassBuilderImpl.AddFunctionMember(nam, FunctionDistiller<F>::Get(), stubFP, stubCtx, params, modifiers);
   return *this;
}


//______________________________________________________________________________
template <typename TD> inline Reflex::ClassBuilder&
Reflex::ClassBuilder::AddTypedef(const char* def) {
   fClassBuilderImpl.AddTypedef(TypeDistiller<TD>::Get(), def);
   return *this;
}


//______________________________________________________________________________
template <typename E> inline Reflex::ClassBuilder&
Reflex::ClassBuilder::AddEnum(const char* values,
                              unsigned int modifiers) {
   fClassBuilderImpl.AddEnum(Tools::Demangle(typeid(E)).c_str(), values, &typeid(E), modifiers);
   return *this;
}


//______________________________________________________________________________
template <typename P> inline Reflex::ClassBuilder&
Reflex::ClassBuilder::AddProperty(const char* key,
                                  P value) {
   fClassBuilderImpl.AddProperty(key, value);
   return *this;
}


//______________________________________________________________________________
template <typename C> inline
Reflex::ClassBuilderT<C>::ClassBuilderT(unsigned int modifiers, TYPE typ):
   fClassBuilderImpl(Tools::Demangle(typeid(C)).c_str(), typeid(C), sizeof(C), modifiers, typ) {
}


//______________________________________________________________________________
template <class C> inline
Reflex::ClassBuilderT<C>::ClassBuilderT(const char* nam, unsigned int modifiers, TYPE typ):
   fClassBuilderImpl(nam, typeid(C), sizeof(C), modifiers, typ) {
}


//______________________________________________________________________________
template <typename C> template <typename B> inline
Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddBase(unsigned int modifiers) {
   fClassBuilderImpl.AddBase(GetType<B>(), BaseOffset<C, B>::Get(), modifiers);
   return *this;
}


//______________________________________________________________________________
template <class C> inline
Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddBase(const Type& bas,
           OffsetFunction offsFP,
           unsigned int modifiers) {
   fClassBuilderImpl.AddBase(bas, offsFP, modifiers);
   return *this;
}


//______________________________________________________________________________
template <class C> template <class T> inline
Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddDataMember(const char* nam,
                 size_t offs,
                 unsigned int modifiers) {
   fClassBuilderImpl.AddDataMember(nam, TypeDistiller<T>::Get(), offs, modifiers);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C> inline
Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddDataMember(const Type& typ,
                 const char* nam,
                 size_t offs,
                 unsigned int modifiers) {
   fClassBuilderImpl.AddDataMember(nam, typ, offs, modifiers);
   return *this;
}


//______________________________________________________________________________
template <typename C> template <typename F> inline 
Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddFunctionMember(const char* nam,
                     StubFunction stubFP,
                     void* stubCtx,
                     const char* params,
                     unsigned int modifiers) {
   fClassBuilderImpl.AddFunctionMember(nam, FunctionDistiller<F>::Get(), stubFP, stubCtx, params, modifiers);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddFunctionMember(const Type& typ,
                     const char* nam,
                     StubFunction stubFP,
                     void* stubCtx,
                     const char* params,
                     unsigned int modifiers) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddFunctionMember(nam,
                                       typ,
                                       stubFP,
                                       stubCtx,
                                       params,
                                       modifiers);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C> template <typename TD>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddTypedef(const char* def) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddTypedef(TypeDistiller<TD>::Get(),
                                def);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddTypedef(const char* typ,
              const char* def) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddTypedef(TypeBuilder(typ),
                                def);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddTypedef(const Type& typ,
              const char* def) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddTypedef(typ,
                                def);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C> template <typename E>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddEnum(const char* values,
           unsigned int modifiers) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddEnum(Tools::Demangle(typeid(E)).c_str(),
                             values,
                             &typeid(E),
                             modifiers);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddEnum(const char* nam,
           const char* values,
           const std::type_info* ti,
           unsigned int modifiers) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddEnum(nam,
                             values,
                             ti,
                             modifiers);
   return *this;
}


/*/-------------------------------------------------------------------------------
   template < class C >
   inline Reflex::ClassBuilderT<C> &
   Reflex::ClassBuilderT<C>::addUnion( const char * nam,
   const char * values,
   unsigned int modifiers ) {
   //-------------------------------------------------------------------------------
   fClassBuilderImpl.addUnion( nam, values, modifiers );
   return * this;
   }
 */


//-------------------------------------------------------------------------------
template <class C> template <class P>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::AddProperty(const char* key,
               P value) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::EnableCallback(bool enable /* = true */) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.EnableCallback(enable);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C>
inline Reflex::ClassBuilderT<C>&
Reflex::ClassBuilderT<C>::SetSizeOf(size_t size) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.SetSizeOf(size);
   return *this;
}


//-------------------------------------------------------------------------------
template <class C> inline Reflex::Type
Reflex::ClassBuilderT<C>::ToType() {
//-------------------------------------------------------------------------------
   return fClassBuilderImpl.ToType();
}


//-------------------------------------------------------------------------------
template <class C> inline void
Reflex::ClassBuilderT<C>::AddOnDemandDataMemberBuilder(OnDemandBuilderForScope* odb) {
//-------------------------------------------------------------------------------
// Register an on demand builder with this class.
   fClassBuilderImpl.AddOnDemandDataMemberBuilder(odb);
}

//-------------------------------------------------------------------------------
template <class C> inline void
Reflex::ClassBuilderT<C>::AddOnDemandFunctionMemberBuilder(OnDemandBuilderForScope* odb) {
//-------------------------------------------------------------------------------
// Register an on demand builder with this class.
   fClassBuilderImpl.AddOnDemandFunctionMemberBuilder(odb);
}

#endif // Reflex_ClassBuilder
