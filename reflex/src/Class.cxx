// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Class.h"

#include "Reflex/Object.h"
#include "Reflex/Type.h"
#include "Reflex/Base.h"

#include "DataMember.h"
#include "FunctionMember.h"
#include "Reflex/Tools.h"

#include <typeinfo>
#include <iostream>
#include <algorithm>
#if defined (__linux) || defined (__APPLE__)
#include <cxxabi.h>
#endif

//-------------------------------------------------------------------------------
ROOT::Reflex::Class::Class(  const char *           typ, 
                             size_t                 size,
                             const std::type_info & ti,
                             unsigned int           modifiers,
                             TYPE                   classType )
//-------------------------------------------------------------------------------
  : TypeBase( typ, size, classType, ti ),
    ScopeBase( typ, classType ),
    fModifiers( modifiers ),
    fAllBases( 0 ),
    fCompleteType( false ),
    fConstructors( std::vector< Member >()),
    fDestructor( Member()),
    fPathsToBase( PathsToBase()) {}
    

//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::AddBase( const Type &   bas,
                                   OffsetFunction offsFP,
                                   unsigned int   modifiers ) const {
//-------------------------------------------------------------------------------
  Base b( bas, offsFP, modifiers );
  fBases.push_back( b );
}

    
//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::Class::CastObject( const Type & to, 
                                                      const Object & obj ) const {
//-------------------------------------------------------------------------------
  std::vector< Base > path = std::vector< Base >();
  if ( HasBase( to, path )) { // up cast 
    // in case of up cast the Offset has to be calculated by Reflex
    size_t obj2 = (size_t)obj.AddressGet();
    for( std::vector< Base >::reverse_iterator bIter = path.rbegin();
         bIter != path.rend(); ++bIter ) {
      obj2 += bIter->Offset((void*)obj2);
    }
    return Object(to,(void*)obj2);
  }
  path.clear();
  Type t = *this;
  if ( to.HasBase( t ) ) {  // down cast
  // use the internal dynamic casting of the compiler (e.g. libstdc++.so)
    void * obj3 = 0;
#if defined (__linux) || defined (__APPLE__)
    obj3 = abi::__dynamic_cast(obj.AddressGet(),
                               (const abi::__class_type_info*)&this->TypeInfo(),
                               (const abi::__class_type_info*)&to.TypeInfo(),
                               -1); 
#elif defined (_WIN32)
    obj3 = __RTDynamicCast(obj.AddressGet(),
                           0,
                           (void*)&this->TypeInfo(),
                           (void*)&to.TypeInfo(),
                           0);
#endif
    return Object(to, obj3);
  }
  // fixme cross cast missing ?? internal cast possible ??

  // if the same TypeNth was passed return the object
  if ((Type)(*this) == to) return obj;

  // if everything fails return the dummy object
  return Object();
}

    
/*/-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::Class::Construct( const Type & signature, 
                                                     std::vector < Object > args, 
                                                     void * mem ) const {
//------------------------------------------------------------------------------- 
  static Type defSignature = Type::ByName("void (void)");
  Type signature2 = signature;
  
  Member constructor = Member();
  if ( !signature &&  fConstructors.size() > 1 ) 
    signature2 = defSignature; 
  
  for (size_t i = 0; i < fConstructors.size(); ++ i) {
    if ( !signature2 || fConstructors[i].TypeGet().Id() == signature2.Id()) {
      constructor = fConstructors[i];
      break;
    }
  }
  
  if ( constructor.TypeGet() ) {
    // no memory AddressGet passed -> Allocate memory for class
    if ( mem == 0 ) mem = Allocate();
    Object obj = Object( TypeGet(), mem );
    constructor.Invoke( obj, args );
    return obj;
  }
  else {
    throw RuntimeError("No suitable constructor found");
  }
}
*/

    
//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::Class::Construct( const Type & signature, 
                                                     std::vector < void * > args, 
                                                     void * mem ) const {
//------------------------------------------------------------------------------- 
  static Type defSignature = Type::ByName("void (void)");
  Type signature2 = signature;
  
  Member constructor = Member();
  if ( !signature &&  fConstructors.size() > 1 ) 
    signature2 = defSignature; 
  
  for (size_t i = 0; i < fConstructors.size(); ++ i) {
    if ( !signature2 || fConstructors[i].TypeGet().Id() == signature2.Id()) {
      constructor = fConstructors[i];
      break;
    }
  }
  
  if ( constructor.TypeGet() ) {
    // no memory AddressGet passed -> Allocate memory for class
    if ( mem == 0 ) mem = Allocate();
    Object obj = Object( TypeGet(), mem );
    constructor.Invoke( obj, args );
    return obj;
  }
        
  throw RuntimeError("No suitable constructor found");
}

    
//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::Destruct( void * instance, 
                                    bool dealloc ) const {
//-------------------------------------------------------------------------------
  if ( ! fDestructor.TypeGet() ) {
    // destructor for this class not yet revealed
    for ( size_t i = 0; i < ScopeBase::FunctionMemberCount(); ++i ) {
      Member fm = ScopeBase::FunctionMemberNth( i );
      // constructor found Set the cache pointer
      if ( fm.IsDestructor() ) {
        fDestructor = fm; 
        break;
      }
    }
  }
  if ( fDestructor.TypeGet()) {
    // we found a destructor -> Invoke it
    Object dummy = Object(Type(), instance);
    fDestructor.Invoke( dummy );
    // if deallocation of memory wanted 
    if ( dealloc ) { Deallocate( instance ); }
  }
  else {
    // this class has no destructor defined we call the operator delete on it
    ::operator delete(instance);
   }
}


//-------------------------------------------------------------------------------
struct DynType {
//-------------------------------------------------------------------------------
  virtual ~DynType() {}
};

    
//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Class::DynamicType( const Object & obj ) const {
//-------------------------------------------------------------------------------
  // If no virtual_function_table return itself
  if ( IsVirtual() ) {
    // Avoid the case that the first word is a virtual_base_offset_table instead of
    // a virtual_function_table  
    long Offset = **(long**)obj.AddressGet();
    if ( Offset == 0 ) return * this;
    else {
      Type dytype = Type::ByTypeInfo(typeid(*(DynType*)obj.AddressGet()));
      if ( dytype && dytype.IsClass() ) return dytype;
      else                              return * this;
    }
  }
  else {
    return * this; 
  }
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::Class::HasBase( const Type & cl ) const {
//-------------------------------------------------------------------------------
  std::vector<Base> v = std::vector<Base>();
  return HasBase(cl, v);
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::Class::HasBase( const Type & cl,  
                                   std::vector< Base > & path ) const {
//-------------------------------------------------------------------------------
  for ( size_t i = 0; i < BaseCount(); ++i ) {
    // is the final BaseNth class one of the current class ?
    if ( BaseNth( i ).ToType().Id() == cl.Id() ) { 
      // remember the path to this class
      path.push_back( BaseNth( i )); 
      return true; 
    }
    // if searched BaseNth class is not direct BaseNth look in the bases of this one
    else if ( BaseNth( i ) && BaseNth( i ).BaseClass()->HasBase( cl, path )) {
      // if successfull remember path
      path.push_back( BaseNth( i )); 
      return true; 
    }
  }
  return false;
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::Class::IsComplete() const {
//-------------------------------------------------------------------------------
  if ( ! fCompleteType ) fCompleteType = IsComplete2(); 
  return fCompleteType;
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::Class::IsComplete2() const {
//-------------------------------------------------------------------------------
  for (size_t i = 0; i < BaseCount(); ++i) {
    Type baseType = BaseNth( i ).ToType();
    if ( ! baseType )  return false;
    if ( ! baseType.IsComplete()) return false;
  }
  return true;
}

    
//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Class::AllBases() const {
//-------------------------------------------------------------------------------
  size_t aBases = 0;
  for ( size_t i = 0; i < BaseCount(); ++i ) {
    ++aBases;
    if ( BaseNth( i )) { 
      aBases += BaseNth( i ).BaseClass()->AllBases();
    }
  }
  return aBases;
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::Class::NewBases() const {
//-------------------------------------------------------------------------------
  if ( ! fCompleteType ) {
    size_t numBases = AllBases();
    if ( fAllBases != numBases ) {
      fCompleteType = IsComplete2();
      fAllBases = numBases;
      return true;
    }
  }
  return false;
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::UpdateMembers() const {
//-------------------------------------------------------------------------------
  std::vector < OffsetFunction > basePath = std::vector < OffsetFunction >();
  UpdateMembers2( fMembers, 
                  fDataMembers, 
                  fFunctionMembers,
                  fPathsToBase,
                  basePath );
}

    
//-------------------------------------------------------------------------------
const std::vector < ROOT::Reflex::OffsetFunction > & 
ROOT::Reflex::Class::PathToBase( const Scope & bas ) const {
//-------------------------------------------------------------------------------
  std::vector < OffsetFunction > * PathToBase = fPathsToBase[ bas.Id() ];
  if ( ! PathToBase ) {
    UpdateMembers();
    PathToBase = fPathsToBase[ bas.Id() ];
    /* fixme can Get rid of UpdateMembers() ?
    std::cerr << Reflex::Argv0() << ": WARNING: No path found from " 
              << this->Name() << " to " << bas.Name() << std::endl;
    if ( NewBases()) {
      std::cerr << Reflex::Argv0() << ": INFO: Not all base classes have resolved, "
                << "do Class::UpdateMembers() and try again " << std::endl; 
    }
    */
  }
  return * PathToBase;
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::UpdateMembers2( Members & members,
                                          Members & dataMembers,
                                          Members & functionMembers,
                                          PathsToBase & pathsToBase,
                                          std::vector < OffsetFunction > & basePath ) const {
//-------------------------------------------------------------------------------
  std::vector < Base >::const_iterator bIter;
  for ( bIter = fBases.begin(); bIter != fBases.end(); ++bIter ) {
    Type bType = bIter->ToType();
    basePath.push_back( bIter->OffsetFP());
    if ( bType ) {
    pathsToBase[ (dynamic_cast<const Class*>(bType.TypeBaseNth()))->ScopeGet().Id() ] = new std::vector < OffsetFunction >( basePath );
      size_t i = 0;
      for ( i = 0; i < bType.DataMemberCount(); ++i ) {
        Member dm = bType.DataMemberNth(i);
        if ( std::find( dataMembers.begin(),
                        dataMembers.end(),
                        dm ) == dataMembers.end()) {
          members.push_back( dm );
          dataMembers.push_back( dm );
        }
      }
      for ( i = 0; i < bType.FunctionMemberCount(); ++i ) {
        Member fm = bType.FunctionMemberNth( i );
        if ( std::find( functionMembers.begin(), 
                        functionMembers.end(),
                        fm ) == functionMembers.end()) {
          members.push_back( fm );
          functionMembers.push_back( fm );
        }
      }
      if ( bType ) (dynamic_cast<const Class*>(bType.TypeBaseNth()))->UpdateMembers2( members,
                                                                                   dataMembers, 
                                                                                   functionMembers,
                                                                                   pathsToBase,
                                                                                   basePath );
    }
    basePath.pop_back();
  }
  /*
  // breath first search to find the "lowest" members in the hierarchy
  for ( bIter = fBases.begin(); bIter != fBases.end(); ++bIter ) {
  const Class * bClass = (*bIter)->toClass();
  if ( bClass ) {  bClass->UpdateMembers2( members,
  dataMembers, 
  functionMembers,
  pathsToBase,
  basePath );
  }
  }
  */
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddDataMember( dm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::AddDataMember( const char * nam,
                                         const Type & typ,
                                         size_t offs,
                                         unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddDataMember( nam, typ, offs, modifiers );
}

    
//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::RemoveDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
  ScopeBase::RemoveDataMember( dm );
}

    
//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::AddFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddFunctionMember( fm );
  if ( fm.IsConstructor() )    fConstructors.push_back( fm );
  else if ( fm.IsDestructor() ) fDestructor = fm;
}

    
//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::AddFunctionMember( const char * nam,
                                             const Type & typ,
                                             StubFunction stubFP,
                                             void * stubCtx,
                                             const char * params,
                                             unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddFunctionMember(nam,typ,stubFP,stubCtx,params,modifiers);
  if ( 0 !=  (modifiers & CONSTRUCTOR )) fConstructors.push_back(fFunctionMembers[fFunctionMembers.size()-1]);
  // setting the destructor is not needed because it is always provided when building the class
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Class::RemoveFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
  ScopeBase::RemoveFunctionMember( fm );
}
