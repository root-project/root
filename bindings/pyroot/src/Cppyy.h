#ifndef PYROOT_CPPYY_H
#define PYROOT_CPPYY_H

#include <string>
#include <stddef.h>

#include "TClassRef.h"

namespace Cppyy {

   typedef ptrdiff_t   TCppScope_t;
   typedef TCppScope_t TCppType_t;
   typedef ptrdiff_t   TCppObject_t;
   typedef ptrdiff_t   TCppMethod_t;
   typedef Long_t      TCppIndex_t;
   typedef void* (*TCppMethPtrGetter_t)( TCppObject_t );

   // temp
   TClassRef& type_from_handle( TCppScope_t handle );
   // -- temp

// name to opaque C++ scope representation -----------------------------------
   TCppIndex_t GetNumScopes( TCppScope_t parent );
   std::string GetScopeName( TCppScope_t parent, TCppIndex_t iscope );
   std::string ResolveName( const std::string& cppitem_name );
   TCppScope_t GetScope( const std::string& scope_name );
   TCppType_t  GetTemplate( const std::string& template_name );
   TCppType_t  GetActualClass( TCppType_t klass, TCppObject_t obj );

// memory management ---------------------------------------------------------
   TCppObject_t Allocate( TCppType_t type );
   void         Deallocate( TCppType_t type, TCppObject_t self );
   void         Destruct( TCppType_t type, TCppObject_t self );

// method/function dispatching -----------------------------------------------
   void         CallV( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   UChar_t      CallB( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   Char_t       CallC( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   Short_t      CallH( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   Int_t        CallI( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   Long_t       CallL( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   Long64_t     CallLL( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   Float_t      CallF( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   Double_t     CallD( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   void*        CallR( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   Char_t*      CallS( TCppMethod_t method, TCppObject_t self, int nargs, void* args );
   TCppObject_t CallConstructor( TCppMethod_t method, TCppType_t klass, int nargs, void* args );
   TCppObject_t CallO( TCppMethod_t method, TCppObject_t self, int nargs, void* args, TCppType_t result_type );

   TCppMethPtrGetter_t GetMethPtrGetter( TCppScope_t scope, TCppIndex_t imeth );

// handling of function argument buffer --------------------------------------
   void*  AllocateFunctionArgs( size_t nargs );
   void   DeallocateFunctionArgs( void* args );
   size_t GetFunctionArgSizeof();
   size_t GetFunctionArgTypeoffset();

// scope reflection information ----------------------------------------------
   Bool_t IsNamespace( TCppScope_t scope );
   Bool_t IsEnum( const std::string& type_name );

// class reflection information ----------------------------------------------
   std::string GetFinalName( TCppType_t type );
   std::string GetScopedFinalName( TCppType_t type );
   Bool_t      HasComplexHierarchy( TCppType_t type );
   TCppIndex_t GetNumBases( TCppType_t type );
   std::string GetBaseName( TCppType_t type, TCppIndex_t ibase );
   Bool_t      IsSubtype( TCppType_t derived, TCppType_t base );

// calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0
   size_t GetBaseOffset( TCppType_t derived, TCppType_t base, TCppObject_t address, int direction );

// method/function reflection information ------------------------------------
   TCppIndex_t  GetNumMethods( TCppScope_t scope );
   TCppIndex_t  GetMethodIndexAt( TCppScope_t scope, TCppIndex_t imeth );
   TCppIndex_t* GetMethodIndicesFromName( TCppScope_t scope, const std::string& name );

   std::string GetMethodName( TCppScope_t scope, TCppIndex_t imethx );
   std::string GetMethodResultType( TCppScope_t scope, TCppIndex_t imeth );
   TCppIndex_t GetMethodNumArgs( TCppScope_t scope, TCppIndex_t imeth );
   TCppIndex_t GetMethodReqArgs( TCppScope_t scope, TCppIndex_t imeth );
   std::string GetMethodArgType( TCppScope_t scope, TCppIndex_t imeth, int iarg );
   std::string GetMethodArgDefault( TCppScope_t scope, TCppIndex_t imeth, int iarg );
   std::string GetMethodSignature( TCppScope_t scope, TCppIndex_t imeth );

   Bool_t      IsMethodTemplate( TCppScope_t scope, TCppIndex_t imeth );
   TCppIndex_t GetMethodNumTemplateArgs( TCppScope_t scope, TCppIndex_t imeth );
   std::string GetMethodTemplateArgName( TCppScope_t scope, TCppIndex_t imeth, TCppIndex_t iarg );

   TCppMethod_t GetMethod( TCppScope_t scope, TCppIndex_t imeth );
   TCppIndex_t  GetGlobalOperator(
      TCppType_t scope, TCppType_t lc, TCppScope_t rc, const std::string& op );

// method properties ---------------------------------------------------------
   Bool_t IsConstructor( TCppType_t type, TCppIndex_t imeth );
   Bool_t IsPublicMethod( TCppType_t type, TCppIndex_t imeth );
   Bool_t IsStaticMethod( TCppType_t type, TCppIndex_t imeth );

// data member reflection information ----------------------------------------
   TCppIndex_t GetNumDatamembers( TCppScope_t scope );
   std::string GetDatamemberName( TCppScope_t scope, TCppIndex_t idata );
   std::string GetDatamemberType( TCppScope_t scope, TCppIndex_t idata );
   ptrdiff_t   GetDatamemberOffset( TCppScope_t scope, TCppIndex_t idata );
   TCppIndex_t GetDatamemberIndex( TCppScope_t scope, const std::string& name );

// data member properties ----------------------------------------------------
   Bool_t IsPublicData( TCppScope_t type, TCppIndex_t idata );
   Bool_t IsStaticData( TCppScope_t type, TCppIndex_t idata );
   Bool_t IsEnumData( TCppScope_t type, TCppIndex_t idata );

} // namespace Cppyy

#endif // ifndef PYROOT_CPPYY_H
