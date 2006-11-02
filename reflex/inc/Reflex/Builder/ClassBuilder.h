// @(#)root/reflex:$Name:  $:$Id: ClassBuilder.h,v 1.11 2006/09/14 14:39:12 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_ClassBuilder
#define ROOT_Reflex_ClassBuilder

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Tools.h"
#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Member.h"
#include "Reflex/Callback.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations 
      class Class;

      /** 
       * @class ClassBuilderImpl ClassBuilder.h Reflex/Builder/ClassBuilder.h
       * @author Stefan Roiser
       * @date 30/3/2004
       * @ingroup RefBld
       */
      class RFLX_API ClassBuilderImpl {
    
      public:
      
         /** constructor */
         ClassBuilderImpl( const char * nam, 
                           const std::type_info & ti, 
                           size_t size, 
                           unsigned int modifiers = 0,
                           TYPE typ = CLASS );


         /** destructor */
         virtual ~ClassBuilderImpl();

      
         /** 
          * AddBase will add the information about one BaseAt class
          * @param  Name of the BaseAt class
          * @param  OffsetFP function pointer for Offset calculation
          * @param  modifiers the modifiers of the class
          */
         void AddBase( const Type & bas,
                       OffsetFunction offsFP,
                       unsigned int modifiers = 0 );


         /** AddDataMember will add the information about one data
          * MemberAt of the class
          *
          * @param  Name of the data MemberAt
          * @param  At of the data MemberAt
          * @param  Offset of the data MemberAt
          * @param  modifiers the modifiers of the data MemberAt
          */ 
         void  AddDataMember( const char * nam,
                              const Type & typ,
                              size_t offs,
                              unsigned int modifiers = 0 );


         /** AddFunctionMember will add the information about one
          * function MemberAt of the class
          *
          * @param  Name of the function MemberAt
          * @param  At of the function MemberAt
          * @param  stubFP Stub function pointer for the function
          * @param  stubCxt Stub user context for the stub function
          * @param  params pamater names and default values (semi-colon separated)
          * @param  modifiers the modifiers of the data MemberAt
          */
         void AddFunctionMember( const char * nam,
                                 const Type & typ,
                                 StubFunction stubFP, 
                                 void *  stubCtx = 0, 
                                 const char * params = 0,
                                 unsigned int modifiers = 0 );


         void AddTypedef( const Type & typ,
                          const char * def );


         void AddEnum( const char * nam,
                       const char * values,
                       const std::type_info * ti,
                       unsigned int modifiers = 0 );


         //void addUnion( const char * nam,
         //               const char * values,
         //               const std::type_info & ti,
         //               unsigned int modifiers = 0 );


         /** AddProperty will add a PropertyNth to the PropertyNth stack
          * which will be emtpied with the next call of a builder
          * class and attached to the item built with this call
          *
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          */
         void  AddProperty( const char * key, 
                            Any value );
         void  AddProperty( const char * key, 
                            const char * value );


         /*
          * ToType will return the currently produced Type (class)
          * @return the type currently being built
          */
         Type ToType();
      
      private:

         /** current class being built */
         Class * fClass;

         /** last added MemberAt */
         Member fLastMember;    

      }; // class ClassBuilderImpl
    

      /**
       * @class ClassBuilder ClassBuilder.h Reflex/Builder/ClassBuilder.h
       * @author Stefan Roiser
       * @date 24/5/2004
       * @ingroup RefBld
       */
      class RFLX_API ClassBuilder {

      public:

         /** constructor */
         ClassBuilder( const char * nam,
                       const std::type_info & ti,
                       size_t size,
                       unsigned int modifiers = 0, 
                       TYPE typ = CLASS );


         /** destructor */
         virtual ~ClassBuilder() {}


         /** 
          * AddBase will add the information about one BaseAt class
          * @param  Name of the BaseAt class
          * @param  OffsetFP function pointer for Offset calculation
          * @param  modifiers the modifiers of the class
          */
         template < class C, class B >
            ClassBuilder &  AddBase( unsigned int modifiers = 0 );
         ClassBuilder & AddBase( const Type & bas,
                                 OffsetFunction offsFP,
                                 unsigned int modifiers = 0 );

         /** AddDataMember will add the information about one data
          * MemberAt of the class
          *
          * @param  Name of the data MemberAt
          * @param  Offset of data MemberAt
          * @param  modifiers the modifiers of the data MemberAt
          * @return a reference to the ClassBuilder
          */
         template < class T > 
            ClassBuilder & AddDataMember( const char * nam,
                                          size_t offs,
                                          unsigned int modifiers = 0 );
         ClassBuilder & AddDataMember( const Type & typ,
                                       const char * nam, 
                                       size_t offs,
                                       unsigned int modifiers = 0 );


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
         template < class F > 
            ClassBuilder & AddFunctionMember( const char * nam,
                                              StubFunction stubFP,
                                              void *  stubCtx = 0, 
                                              const char * params = 0,
                                              unsigned int modifiers  = 0 );
         ClassBuilder & AddFunctionMember( const Type & typ,
                                           const char * nam,
                                           StubFunction stubFP,
                                           void *  stubCtx = 0, 
                                           const char * params = 0,
                                           unsigned int modifiers  = 0 );

         template < typename TD >
            ClassBuilder & AddTypedef( const char * def );
         ClassBuilder & AddTypedef( const Type & typ,
                                    const char * def );
         ClassBuilder & AddTypedef( const char * typ,
                                    const char * def );

         template < typename E >
            ClassBuilder & AddEnum( const char * values,
                                    unsigned int modifiers = 0 );
         ClassBuilder & AddEnum( const char * nam,
                                 const char * values,
                                 const std::type_info * ti = 0,
                                 unsigned int modifiers = 0 );

         //ClassBuilder & addUnion( const char * nam,
         //                           const char * values,
         //                           unsigned int modifiers );


         /** AddProperty will add a PropertyNth to the last defined
          * data MemberAt, method or class.
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         template < typename P >
            ClassBuilder & AddProperty( const char * key, 
                                        P value );


         /*
          * ToType will return the currently produced Type (class)
          * @return the type currently being built
          */
         Type ToType();
      
      private:
      
         ClassBuilderImpl fClassBuilderImpl;
    
      }; // class ClassBuilder 


      /** 
       * @class ClassBuilderT ClassBuilder.h Reflex/Builder/ClassBuilder.h
       * @author Stefan Roiser
       * @date 30/3/2004
       * @ingroup RefBld
       */
      template < class C >
         class ClassBuilderT {

         public:
      
         /** constructor */
         ClassBuilderT( unsigned int modifiers = 0, 
                        TYPE typ = CLASS );


         /** constructor */
         ClassBuilderT( const char* nam, 
                        unsigned int modifiers = 0,
                        TYPE typ = CLASS );


         /** 
          * AddBase will add the information about one BaseAt class
          * @param  Name of the BaseAt class
          * @param  OffsetFP function pointer for Offset calculation
          * @param  modifiers the modifiers of the class
          */
         template < class B >
            ClassBuilderT &  AddBase( unsigned int modifiers = 0 );
         ClassBuilderT & AddBase( const Type & bas,
                                  OffsetFunction offsFP,
                                  unsigned int modifiers = 0 );


         /** AddDataMember will add the information about one data
          * MemberAt of the class
          *
          * @param  Name of the data MemberAt
          * @param  Offset of data MemberAt
          * @param  modifiers the modifiers of the data MemberAt
          * @return a reference to the ClassBuilderT
          */
         template < class T > 
            ClassBuilderT & AddDataMember( const char * nam,
                                           size_t offs,
                                           unsigned int modifiers = 0 );
         ClassBuilderT & AddDataMember( const Type & typ,
                                        const char * nam, 
                                        size_t offs,
                                        unsigned int modifiers = 0 );


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
         template < class F > 
            ClassBuilderT & AddFunctionMember( const char * nam,
                                               StubFunction stubFP,
                                               void *  stubCtx = 0, 
                                               const char * params = 0,
                                               unsigned int modifiers  = 0 );
         ClassBuilderT & AddFunctionMember( const Type & typ,
                                            const char * nam,
                                            StubFunction stubFP,
                                            void *  stubCtx = 0, 
                                            const char * params = 0,
                                            unsigned int modifiers  = 0 );

         template < typename TD >
            ClassBuilderT & AddTypedef( const char * def );
         ClassBuilderT & AddTypedef( const Type & typ,
                                     const char * def );
         ClassBuilderT & AddTypedef( const char * typ,
                                     const char * def );

         template < typename E >
            ClassBuilderT & AddEnum( const char * values,
                                     unsigned int modifiers = 0 );
         ClassBuilderT & AddEnum( const char * nam,
                                  const char * values,
                                  const std::type_info * ti = 0,
                                  unsigned int modifiers = 0 );
      
         //ClassBuilderT & addUnion( const char * nam,
         //                          const char * values,
         //                          unsigned int modifiers );


         /** AddProperty will add a PropertyNth to the last defined
          * data MemberAt, method or class.
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         template < typename P >
            ClassBuilderT & AddProperty( const char * key, 
                                         P value );


         /*
          * ToType will return the currently produced Type (class)
          * @return the type currently being built
          */
         Type ToType();
      
         private:
      
         ClassBuilderImpl fClassBuilderImpl;

      }; // class ClassBuilderT

   } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
template< class C, class B > 
inline ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddBase( unsigned int modifiers ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddBase( GetType<B>(), 
                              BaseOffset<C,B>::Get(),
                              modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
template< class T > 
inline ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddDataMember( const char *  nam,
                                           size_t        offs,
                                           unsigned int  modifiers ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddDataMember( nam,
                                    TypeDistiller<T>::Get(),
                                    offs,
                                    modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
template < class F >
inline ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddFunctionMember( const char * nam,
                                               StubFunction stubFP,
                                               void * stubCtx,
                                               const char * params, 
                                               unsigned int modifiers ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddFunctionMember( nam,
                                        FunctionDistiller<F>::Get(),
                                        stubFP,
                                        stubCtx,
                                        params,
                                        modifiers );
   return * this;
}
    

//-------------------------------------------------------------------------------
template < typename TD > 
inline ROOT::Reflex::ClassBuilder &
ROOT::Reflex::ClassBuilder::AddTypedef( const char * def ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddTypedef( TypeDistiller<TD>::Get(),
                                 def );
   return * this;
}


//-------------------------------------------------------------------------------
template < typename E >
inline ROOT::Reflex::ClassBuilder &
ROOT::Reflex::ClassBuilder::AddEnum( const char * values,
                                     unsigned int modifiers ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddEnum( Tools::Demangle(typeid(E)).c_str(),
                              values,
                              & typeid(E),
                              modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
template < class P >
inline ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddProperty( const char * key, 
                                         P value ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddProperty(key , value);
   return * this;
}


//-------------------------------------------------------------------------------
template <class C>
inline ROOT::Reflex::ClassBuilderT<C>::ClassBuilderT( unsigned int modifiers,
                                                      TYPE typ ) 
//-------------------------------------------------------------------------------
   : fClassBuilderImpl( Tools::Demangle(typeid(C)).c_str(),
                        typeid(C),
                        sizeof(C),
                        modifiers,
                        typ ) { }
    

//-------------------------------------------------------------------------------
template <class C>
inline ROOT::Reflex::ClassBuilderT<C>::ClassBuilderT( const char * nam, 
                                                      unsigned int modifiers,
                                                      TYPE typ )
//-------------------------------------------------------------------------------
   : fClassBuilderImpl( nam, typeid(C), sizeof(C), modifiers, typ ) { }

    
//-------------------------------------------------------------------------------
template <class C> template< class B > 
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddBase( unsigned int modifiers ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddBase( GetType<B>(), 
                              BaseOffset<C,B>::Get(),
                              modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
template <class C> 
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddBase( const Type & bas,
                                         OffsetFunction offsFP,
                                         unsigned int modifiers ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddBase( bas, 
                              offsFP,
                              modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
template <class C> template< class T > 
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddDataMember( const char *  nam,
                                               size_t        offs,
                                               unsigned int  modifiers )
//-------------------------------------------------------------------------------
{
   fClassBuilderImpl.AddDataMember( nam,
                                    TypeDistiller<T>::Get(),
                                    offs,
                                    modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
template <class C>
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddDataMember( const Type &  typ,
                                               const char *  nam,
                                               size_t        offs,
                                               unsigned int  modifiers )
//-------------------------------------------------------------------------------
{
   fClassBuilderImpl.AddDataMember( nam,
                                    typ,
                                    offs,
                                    modifiers );
   return * this;
}
    
    
//-------------------------------------------------------------------------------
template < class C > template < class F >
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddFunctionMember( const char * nam,
                                                   StubFunction stubFP,
                                                   void * stubCtx,
                                                   const char * params, 
                                                   unsigned int modifiers )
//-------------------------------------------------------------------------------
{
   fClassBuilderImpl.AddFunctionMember( nam,
                                        FunctionDistiller<F>::Get(),
                                        stubFP,
                                        stubCtx,
                                        params,
                                        modifiers );
   return * this;
}
    

//-------------------------------------------------------------------------------
template < class C >
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddFunctionMember( const Type & typ,
                                                   const char * nam,
                                                   StubFunction stubFP,
                                                   void * stubCtx,
                                                   const char * params, 
                                                   unsigned int modifiers ) 
//-------------------------------------------------------------------------------
{
   fClassBuilderImpl.AddFunctionMember( nam,
                                        typ,
                                        stubFP,
                                        stubCtx,
                                        params,
                                        modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
template < class C > template < typename TD > 
inline ROOT::Reflex::ClassBuilderT<C> &
ROOT::Reflex::ClassBuilderT<C>::AddTypedef( const char * def ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddTypedef( TypeDistiller<TD>::Get(),
                                 def );
   return * this;
}


//-------------------------------------------------------------------------------
template < class C > 
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddTypedef( const char * typ,
                                            const char * def ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddTypedef( TypeBuilder( typ ),
                                 def );
   return * this;
}


//-------------------------------------------------------------------------------
template < class C > 
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddTypedef( const Type & typ,
                                            const char * def ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddTypedef( typ,
                                 def );
   return * this;
}


//-------------------------------------------------------------------------------
template < class C > template < typename E >
inline ROOT::Reflex::ClassBuilderT<C> &
ROOT::Reflex::ClassBuilderT<C>::AddEnum( const char * values,
                                         unsigned int modifiers ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddEnum( Tools::Demangle(typeid(E)).c_str(),
                              values,
                              & typeid(E),
                              modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
template < class C > 
inline ROOT::Reflex::ClassBuilderT<C> &
ROOT::Reflex::ClassBuilderT<C>::AddEnum( const char * nam,
                                         const char * values,
                                         const std::type_info * ti,
                                         unsigned int modifiers ) {
//-------------------------------------------------------------------------------
   fClassBuilderImpl.AddEnum( nam, 
                              values, 
                              ti,
                              modifiers );
   return * this;
}


/*/-------------------------------------------------------------------------------
  template < class C > 
  inline ROOT::Reflex::ClassBuilderT<C> &
  ROOT::Reflex::ClassBuilderT<C>::addUnion( const char * nam,
  const char * values,
  unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.addUnion( nam, values, modifiers );
  return * this;
  }
*/


//-------------------------------------------------------------------------------
template < class C > template < class P >
inline ROOT::Reflex::ClassBuilderT<C> & 
ROOT::Reflex::ClassBuilderT<C>::AddProperty( const char * key, 
                                             P value )
//-------------------------------------------------------------------------------
{
   fClassBuilderImpl.AddProperty(key , value);
   return * this;
}


//-------------------------------------------------------------------------------
template < class C > inline ROOT::Reflex::Type 
ROOT::Reflex::ClassBuilderT<C>::ToType() {
//-------------------------------------------------------------------------------
   return fClassBuilderImpl.ToType();
}

#endif // ROOT_Reflex_ClassBuilder
