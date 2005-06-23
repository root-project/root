// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_ClassBuilder
#define ROOT_Reflex_ClassBuilder

// Include files
#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "Reflex/Builder/TypeBuilder.h"

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
    class ClassBuilderImpl {
    
    public:
      
      /** constructor */
      ClassBuilderImpl( const char * Name, 
                        const std::type_info &, 
                        size_t size, 
                        unsigned int modifiers = 0 );


      /** destructor */
      virtual ~ClassBuilderImpl();

      
      /** 
       * AddBase will add the information about one BaseNth class
       * @param  Name of the BaseNth class
       * @param  OffsetFP function pointer for Offset calculation
       * @param  modifiers the modifiers of the class
       */
      void AddBase( const Type & BaseNth,
                    OffsetFunction OffsetFP,
                    unsigned int modifiers = 0 );


      /** AddDataMember will add the information about one data
       * MemberNth of the class
       *
       * @param  Name of the data MemberNth
       * @param  TypeNth of the data MemberNth
       * @param  Offset of the data MemberNth
       * @param  modifiers the modifiers of the data MemberNth
       */ 
      void  AddDataMember( const char * Name,
                           const Type & TypeNth,
                           size_t Offset,
                           unsigned int modifiers = 0 );


      /** AddFunctionMember will add the information about one
       * function MemberNth of the class
       *
       * @param  Name of the function MemberNth
       * @param  TypeNth of the function MemberNth
       * @param  stubFP Stub function pointer for the function
       * @param  stubCxt Stub user context for the stub function
       * @param  params pamater names and default values (semi-colon separated)
       * @param  modifiers the modifiers of the data MemberNth
       */
      void AddFunctionMember( const char * Name,
                              const Type & TypeNth,
                              StubFunction stubFP, 
                              void *  stubCtx = 0, 
                              const char * params = 0,
                              unsigned int modifiers = 0 );


      void AddTypedef( const Type & TypeNth,
                       const char * def );


      void AddEnum( const char * Name,
                    const char * values,
                    const std::type_info * typeinfo );


      //void addUnion( const char * Name,
      //               const char * values,
      //               const std::type_info & typeinfo );


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
      
    private:

      /** current class being built */
      Class * fClass;

      /** last added MemberNth */
      Member fLastMember;    

    }; // class ClassBuilderImpl
    

    /**
     * @class ClassBuilderNT ClassBuilder.h Reflex/Builder/ClassBuilder.h
     * @author Stefan Roiser
     * @date 24/5/2004
     * @ingroup RefBld
     */
    class ClassBuilderNT {

    public:

      /** constructor */
      ClassBuilderNT( const char * Name,
		      const std::type_info & TypeInfo,
		      size_t size,
		      unsigned int modifiers = 0 );

      /** destructor */
      virtual ~ClassBuilderNT() {}


      /** 
       * AddBase will add the information about one BaseNth class
       * @param  Name of the BaseNth class
       * @param  OffsetFP function pointer for Offset calculation
       * @param  modifiers the modifiers of the class
       */
      template < class C, class B >
        ClassBuilderNT &  AddBase( unsigned int modifiers = 0 );


      /** AddDataMember will add the information about one data
       * MemberNth of the class
       *
       * @param  Name of the data MemberNth
       * @param  Offset of data MemberNth
       * @param  modifiers the modifiers of the data MemberNth
       * @return a reference to the ClassBuilderNT
       */
      template < class T > 
        ClassBuilderNT & AddDataMember( const char * Name,
                                      size_t Offset,
                                      unsigned int modifiers = 0 );
      ClassBuilderNT & AddDataMember( const Type & TypeNth,
                                    const char * Name, 
                                    size_t Offset,
                                    unsigned int modifiers = 0 );


      /** AddFunctionMember will add the information about one
       * function MemberNth of the class
       *
       * @param  Name of the function MemberNth
       * @param  function templated function MemberNth to extract TypeNth information
       * @param  stubFP Stub function pointer for the function
       * @param  stubCxt Stub user context for the stub function
       * @param  params pamater names and default values (semi-colon separated)
       * @param  modifiers the modifiers of the data MemberNth
       * @return a reference to the ClassBuilderNT
       */
      template < class F > 
        ClassBuilderNT & AddFunctionMember( const char * Name,
                                          StubFunction stubFP,
                                          void *  stubCtx = 0, 
                                          const char * params = 0,
                                          unsigned int modifiers  = 0 );
      ClassBuilderNT & AddFunctionMember( const Type & TypeNth,
                                        const char * Name,
                                        StubFunction stubFP,
                                        void *  stubCtx = 0, 
                                        const char * params = 0,
                                        unsigned int modifiers  = 0 );

      template < typename TD >
        ClassBuilderNT & AddTypedef( const char * def );
      ClassBuilderNT & AddTypedef( const Type & TypeNth,
                                 const char * def );
      ClassBuilderNT & AddTypedef( const char * TypeNth,
                                 const char * def );

      template < typename E >
        ClassBuilderNT & AddEnum( const char * values );
      ClassBuilderNT & AddEnum( const char * Name,
                              const char * values,
                              const std::type_info * typeinfo = 0 );

      //ClassBuilderNT & addUnion( const char * Name,
      //                         const char * values );


      /** AddProperty will add a PropertyNth to the last defined
       * data MemberNth, method or class.
       * @param  key the PropertyNth key
       * @param  value the value of the PropertyNth
       * @return a reference to the building class
       */
      template < typename P >
        ClassBuilderNT & AddProperty( const char * key, 
                                    P value );

    private:
      
      ClassBuilderImpl fClassBuilderImpl;
    
    }; // class ClassBuilderNT 


    /** 
     * @class ClassBuilder ClassBuilder.h Reflex/Builder/ClassBuilder.h
     * @author Stefan Roiser
     * @date 30/3/2004
     * @ingroup RefBld
     */
    template < class C >
      class ClassBuilder {

    public:
      
      /** constructor */
      ClassBuilder( unsigned int modifiers = 0 );


      /** constructor */
      ClassBuilder( const char* Name, 
                    unsigned int modifiers = 0 );


      /** 
       * AddBase will add the information about one BaseNth class
       * @param  Name of the BaseNth class
       * @param  OffsetFP function pointer for Offset calculation
       * @param  modifiers the modifiers of the class
       */
      template < class B >
        ClassBuilder &  AddBase( unsigned int modifiers = 0 );


      /** AddDataMember will add the information about one data
       * MemberNth of the class
       *
       * @param  Name of the data MemberNth
       * @param  Offset of data MemberNth
       * @param  modifiers the modifiers of the data MemberNth
       * @return a reference to the ClassBuilder
       */
      template < class T > 
        ClassBuilder & AddDataMember( const char * Name,
                                      size_t Offset,
                                      unsigned int modifiers = 0 );
      ClassBuilder & AddDataMember( const Type & TypeNth,
                                    const char * Name, 
                                    size_t Offset,
                                    unsigned int modifiers = 0 );


      /** AddFunctionMember will add the information about one
       * function MemberNth of the class
       *
       * @param  Name of the function MemberNth
       * @param  function templated function MemberNth to extract TypeNth information
       * @param  stubFP Stub function pointer for the function
       * @param  stubCxt Stub user context for the stub function
       * @param  params pamater names and default values (semi-colon separated)
       * @param  modifiers the modifiers of the data MemberNth
       * @return a reference to the ClassBuilder
       */
      template < class F > 
        ClassBuilder & AddFunctionMember( const char * Name,
                                          StubFunction stubFP,
                                          void *  stubCtx = 0, 
                                          const char * params = 0,
                                          unsigned int modifiers  = 0 );
      ClassBuilder & AddFunctionMember( const Type & TypeNth,
                                        const char * Name,
                                        StubFunction stubFP,
                                        void *  stubCtx = 0, 
                                        const char * params = 0,
                                        unsigned int modifiers  = 0 );

      template < typename TD >
        ClassBuilder & AddTypedef( const char * def );
      ClassBuilder & AddTypedef( const Type & TypeNth,
                                 const char * def );
      ClassBuilder & AddTypedef( const char * TypeNth,
                                 const char * def );

      template < typename E >
        ClassBuilder & AddEnum( const char * values );
      ClassBuilder & AddEnum( const char * Name,
                              const char * values,
                              const std::type_info * typeinfo = 0 );

      //ClassBuilder & addUnion( const char * Name,
      //                         const char * values );


      /** AddProperty will add a PropertyNth to the last defined
       * data MemberNth, method or class.
       * @param  key the PropertyNth key
       * @param  value the value of the PropertyNth
       * @return a reference to the building class
       */
      template < typename P >
        ClassBuilder & AddProperty( const char * key, 
                                    P value );

    private:
      
      ClassBuilderImpl fClassBuilderImpl;

    }; // class ClassBuilder

  } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassBuilderNT::ClassBuilderNT( const char * Name, 
						 const std::type_info & TypeInfo,
						 size_t size,
						 unsigned int modifiers ) 
//-------------------------------------------------------------------------------
  : fClassBuilderImpl( Name, TypeInfo, size, modifiers ) { }
    

//-------------------------------------------------------------------------------
template< class C, class B > 
inline ROOT::Reflex::ClassBuilderNT & 
ROOT::Reflex::ClassBuilderNT::AddBase( unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddBase( getType<B>(), 
                              baseOffset<C,B>::Get(),
                              modifiers );
  return * this;
}


//-------------------------------------------------------------------------------
template< class T > 
inline ROOT::Reflex::ClassBuilderNT & 
ROOT::Reflex::ClassBuilderNT::AddDataMember( const char *  Name,
					   size_t        Offset,
					   unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddDataMember( Name,
                                    TypeDistiller<T>::Get(),
                                    Offset,
                                    modifiers );
  return * this;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassBuilderNT & 
ROOT::Reflex::ClassBuilderNT::AddDataMember( const Type &  TypeNth,
					   const char *  Name,
					   size_t        Offset,
					   unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddDataMember( Name,
                                    TypeNth,
                                    Offset,
                                    modifiers );
  return * this;
}
    
    
//-------------------------------------------------------------------------------
template < class F >
inline ROOT::Reflex::ClassBuilderNT & 
ROOT::Reflex::ClassBuilderNT::AddFunctionMember( const char * Name,
					       StubFunction stubFP,
					       void * stubCtx,
					       const char * params, 
					       unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddFunctionMember( Name,
                                        FunctionDistiller<F>::Get(),
                                        stubFP,
                                        stubCtx,
                                        params,
                                        modifiers );
  return * this;
}
    

//-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassBuilderNT & 
ROOT::Reflex::ClassBuilderNT::AddFunctionMember( const Type & TypeNth,
					       const char * Name,
					       StubFunction stubFP,
					       void * stubCtx,
					       const char * params, 
					       unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddFunctionMember( Name,
                                        TypeNth,
                                        stubFP,
                                        stubCtx,
                                        params,
                                        modifiers );
  return * this;
}


//-------------------------------------------------------------------------------
template < typename TD > 
inline ROOT::Reflex::ClassBuilderNT &
ROOT::Reflex::ClassBuilderNT::AddTypedef( const char * def ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddTypedef( TypeDistiller<TD>::Get(),
                                 def );
  return * this;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassBuilderNT & 
ROOT::Reflex::ClassBuilderNT::AddTypedef( const char * TypeNth,
                                           const char * def ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddTypedef( TypeBuilder( TypeNth ),
                                 def );
  return * this;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassBuilderNT & 
ROOT::Reflex::ClassBuilderNT::AddTypedef( const Type & TypeNth,
					const char * def ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddTypedef( TypeNth,
                                 def );
  return * this;
}


//-------------------------------------------------------------------------------
template < typename E >
inline ROOT::Reflex::ClassBuilderNT &
ROOT::Reflex::ClassBuilderNT::AddEnum( const char * values ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddEnum( Tools::Demangle(typeid(E)).c_str(),
                              values,
                              & typeid(E));
  return * this;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassBuilderNT &
ROOT::Reflex::ClassBuilderNT::AddEnum( const char * Name,
				     const char * values,
				     const std::type_info * typeinfo ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddEnum( Name, 
                              values, 
                              typeinfo );
  return * this;
}


/*/-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassBuilderNT &
ROOT::Reflex::ClassBuilderNT::addUnion( const char * Name,
                                      const char * values ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.addUnion( Name, values );
  return * this;
}
*/


//-------------------------------------------------------------------------------
template < class P >
inline ROOT::Reflex::ClassBuilderNT & 
ROOT::Reflex::ClassBuilderNT::AddProperty( const char * key, 
					 P value ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddProperty(key , value);
  return * this;
}


//-------------------------------------------------------------------------------
template <class C>
inline ROOT::Reflex::ClassBuilder<C>::ClassBuilder( unsigned int modifiers ) 
//-------------------------------------------------------------------------------
  : fClassBuilderImpl( Tools::Demangle(typeid(C)).c_str(),
                        typeid(C),
                        sizeof(C),
                        modifiers ) { }
    

//-------------------------------------------------------------------------------
template <class C>
inline ROOT::Reflex::ClassBuilder<C>::ClassBuilder( const char * Name, 
                                                    unsigned int modifiers )
//-------------------------------------------------------------------------------
  : fClassBuilderImpl( Name, typeid(C), sizeof(C), modifiers ) { }

    
//-------------------------------------------------------------------------------
template <class C> template< class B > 
inline ROOT::Reflex::ClassBuilder<C> & 
ROOT::Reflex::ClassBuilder<C>::AddBase( unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddBase( getType<B>(), 
                              baseOffset<C,B>::Get(),
                              modifiers );
  return * this;
}


//-------------------------------------------------------------------------------
template <class C> template< class T > 
inline ROOT::Reflex::ClassBuilder<C> & 
ROOT::Reflex::ClassBuilder<C>::AddDataMember( const char *  Name,
                                              size_t        Offset,
                                              unsigned int modifiers )
//-------------------------------------------------------------------------------
{
  fClassBuilderImpl.AddDataMember( Name,
                                    TypeDistiller<T>::Get(),
                                    Offset,
                                    modifiers );
  return * this;
}


//-------------------------------------------------------------------------------
template <class C>
inline ROOT::Reflex::ClassBuilder<C> & 
ROOT::Reflex::ClassBuilder<C>::AddDataMember( const Type &  TypeNth,
                                              const char *  Name,
                                              size_t        Offset,
                                              unsigned int modifiers )
//-------------------------------------------------------------------------------
{
  fClassBuilderImpl.AddDataMember( Name,
                                    TypeNth,
                                    Offset,
                                    modifiers );
  return * this;
}
    
    
//-------------------------------------------------------------------------------
template < class C > template < class F >
inline ROOT::Reflex::ClassBuilder<C> & 
ROOT::Reflex::ClassBuilder<C>::AddFunctionMember( const char * Name,
                                                  StubFunction stubFP,
                                                  void * stubCtx,
                                                  const char * params, 
                                                  unsigned int modifiers )
//-------------------------------------------------------------------------------
{
  fClassBuilderImpl.AddFunctionMember( Name,
                                        FunctionDistiller<F>::Get(),
                                        stubFP,
                                        stubCtx,
                                        params,
                                        modifiers );
  return * this;
}
    

//-------------------------------------------------------------------------------
template < class C >
inline ROOT::Reflex::ClassBuilder<C> & 
ROOT::Reflex::ClassBuilder<C>::AddFunctionMember( const Type & TypeNth,
                                                  const char * Name,
                                                  StubFunction stubFP,
                                                  void * stubCtx,
                                                  const char * params, 
                                                  unsigned int modifiers ) 
//-------------------------------------------------------------------------------
{
  fClassBuilderImpl.AddFunctionMember( Name,
                                        TypeNth,
                                        stubFP,
                                        stubCtx,
                                        params,
                                        modifiers );
  return * this;
}


//-------------------------------------------------------------------------------
template < class C > template < typename TD > 
inline ROOT::Reflex::ClassBuilder<C> &
ROOT::Reflex::ClassBuilder<C>::AddTypedef( const char * def ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddTypedef( TypeDistiller<TD>::Get(),
                                 def );
  return * this;
}


//-------------------------------------------------------------------------------
template < class C > 
inline ROOT::Reflex::ClassBuilder<C> & 
ROOT::Reflex::ClassBuilder<C>::AddTypedef( const char * TypeNth,
                                           const char * def ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddTypedef( TypeBuilder( TypeNth ),
                                 def );
  return * this;
}


//-------------------------------------------------------------------------------
template < class C > 
inline ROOT::Reflex::ClassBuilder<C> & 
ROOT::Reflex::ClassBuilder<C>::AddTypedef( const Type & TypeNth,
                                           const char * def ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddTypedef( TypeNth,
                                 def );
  return * this;
}


//-------------------------------------------------------------------------------
template < class C > template < typename E >
inline ROOT::Reflex::ClassBuilder<C> &
ROOT::Reflex::ClassBuilder<C>::AddEnum( const char * values ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddEnum( Tools::Demangle(typeid(E)).c_str(),
                              values,
                              & typeid(E));
  return * this;
}


//-------------------------------------------------------------------------------
template < class C > 
inline ROOT::Reflex::ClassBuilder<C> &
ROOT::Reflex::ClassBuilder<C>::AddEnum( const char * Name,
                                        const char * values,
                                        const std::type_info * typeinfo ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.AddEnum( Name, 
                              values, 
                              typeinfo );
  return * this;
}


/*/-------------------------------------------------------------------------------
template < class C > 
inline ROOT::Reflex::ClassBuilder<C> &
ROOT::Reflex::ClassBuilder<C>::addUnion( const char * Name,
                                      const char * values ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.addUnion( Name, values );
  return * this;
}
*/


//-------------------------------------------------------------------------------
template < class C > template < class P >
inline ROOT::Reflex::ClassBuilder<C> & 
ROOT::Reflex::ClassBuilder<C>::AddProperty( const char * key, 
                                            P value )
//-------------------------------------------------------------------------------
{
  fClassBuilderImpl.AddProperty(key , value);
  return * this;
}

#endif // ROOT_Reflex_ClassBuilder
