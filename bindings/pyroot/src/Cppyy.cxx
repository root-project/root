#include "PyROOT.h"
#include "Cppyy.h"

// ROOT
#include "TBaseClass.h"
#include "TClass.h"
#include "TClassRef.h"
#include "TClassTable.h"
#include "TClassEdit.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TFunction.h"
#include "TGlobal.h"
#include "TList.h"
#include "TMethod.h"
#include "TROOT.h"

// Standard
#include <assert.h>
#include <map>
#include <sstream>
#include <vector>

// temp
#include <iostream>
// --temp


// data for life time management ---------------------------------------------
typedef std::vector< TClassRef > ClassRefs_t;
static ClassRefs_t g_classrefs( 1 );
static const ClassRefs_t::size_type GLOBAL_HANDLE = 1;

typedef std::map< std::string, ClassRefs_t::size_type > ClassRefIndices_t;
static ClassRefIndices_t g_classref_indices;

typedef std::vector< TFunction > GlobalFuncs_t;
static GlobalFuncs_t g_globalfuncs;

typedef std::vector< TGlobal > GlobalVars_t;
static GlobalVars_t g_globalvars;


// global initialization -----------------------------------------------------
//static inline
TClassRef& Cppyy::type_from_handle( Cppyy::TCppScope_t handle ) {
   assert( (ClassRefs_t::size_type) handle < g_classrefs.size());
   return g_classrefs[(ClassRefs_t::size_type)handle];
}

namespace {

class ApplicationStarter {
public:
   ApplicationStarter() {
      // setup dummy holders for global and std namespaces
      assert( g_classrefs.size() == GLOBAL_HANDLE );
      g_classref_indices[ "" ] = GLOBAL_HANDLE;
      g_classrefs.push_back(TClassRef(""));
      // ROOT ignores std/::std, so point them to the global namespace
      g_classref_indices[ "std" ]   = GLOBAL_HANDLE;
      g_classref_indices[ "::std" ] = GLOBAL_HANDLE;
   }
} _applicationStarter;

} // unnamed namespace


// local helpers -------------------------------------------------------------
// type_from_handle to go here
static inline TFunction* type_get_method( Cppyy::TCppType_t handle, Cppyy::TCppIndex_t idx ) {
   TClassRef& cr = Cppyy::type_from_handle(handle); // FIXME
   if ( cr.GetClass() )
      return (TFunction*)cr->GetListOfMethods()->At( idx );
   assert( handle == (Cppyy::TCppType_t)GLOBAL_HANDLE );
   return (TFunction*)idx;
}


// name to opaque C++ scope representation -----------------------------------
Cppyy::TCppIndex_t Cppyy::GetNumScopes( TCppScope_t handle ) {
   TClassRef& cr = type_from_handle( handle );
   if ( cr.GetClass() ) return 0;   // not supported if not at global scope
   assert( handle == (TCppScope_t)GLOBAL_HANDLE );
   return gClassTable->Classes();
}

std::string Cppyy::GetScopeName( TCppScope_t parent, TCppIndex_t iscope ) {
// Retrieve the scope name of the scope indexed with iscope in parent.
   TClassRef& cr = type_from_handle( parent );
   if ( cr.GetClass() ) return 0;   // not supported if not at global scope
   assert( parent == (TCppScope_t)GLOBAL_HANDLE );
   std::string name = gClassTable->At( iscope );
   if ( name.find("::") == std::string::npos )
       return name;
   return "";
}

std::string Cppyy::ResolveName( const std::string& cppitem_name ) {
// Fully resolve the given name to the final type name.
   std::string tclean = TClassEdit::CleanType( cppitem_name.c_str() );

   TDataType* dt = gROOT->GetType( tclean.c_str() );
   if ( dt ) return dt->GetFullTypeName();
   return TClassEdit::ResolveTypedef( tclean.c_str(), true );
}

Cppyy::TCppScope_t Cppyy::GetScope( const std::string& sname ) {
   std::string scope_name;
   if ( sname.find( "std::", 0, 5 ) == 0 )
      scope_name = sname.substr( 5, std::string::npos );
   else
      scope_name = sname;

   ClassRefIndices_t::iterator icr = g_classref_indices.find( scope_name );
   if ( icr != g_classref_indices.end() )
      return (TCppType_t)icr->second;

   // use TClass directly, to enable auto-loading
   TClassRef cr( TClass::GetClass( scope_name.c_str(), kTRUE, kTRUE ) );
   if ( !cr.GetClass() )
      return (TCppScope_t)NULL;

   // no check for ClassInfo as forward declared classes are okay (fragile)

   ClassRefs_t::size_type sz = g_classrefs.size();
   g_classref_indices[ scope_name ] = sz;
   g_classrefs.push_back( TClassRef( scope_name.c_str() ) );
   return (TCppScope_t)sz;
}

Cppyy::TCppType_t Cppyy::GetTemplate( const std::string& /* template_name */ ) {
   return (TCppType_t)0;
}

Cppyy::TCppType_t Cppyy::GetActualClass( TCppType_t klass, TCppObject_t obj ) {
   TClassRef& cr = type_from_handle( klass );
   TClass* clActual = cr->GetActualClass( (void*)obj );
   if ( clActual && clActual != cr.GetClass() ) {
      // TODO: lookup through name should not be needed
      return (TCppType_t)GetScope( clActual->GetName() );
   }
   return klass;
}


// memory management ---------------------------------------------------------
Cppyy::TCppObject_t Cppyy::Allocate( TCppType_t klass ) {
   TClassRef& cr = type_from_handle( klass );
   return (TCppObject_t)malloc( cr->Size() );
}

void Cppyy::Deallocate( TCppType_t /* klass */, TCppObject_t instance ) {
   free( (void*)instance );
}

void Cppyy::Destruct( TCppType_t klass, TCppObject_t instance ) {
   TClassRef& cr = type_from_handle( klass );
   cr->Destructor( (void*)instance, true );
}


// method/function dispatching -----------------------------------------------
void Cppyy::CallV( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   /* empty */
}

UChar_t Cppyy::CallB( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return (UChar_t)'F';
}

Char_t Cppyy::CallC( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return (Char_t)'F';
}

Short_t Cppyy::CallH( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return (Short_t)0;
}

Int_t Cppyy::CallI( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return 0;
}

Long_t Cppyy::CallL( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return 0;
}

Long64_t Cppyy::CallLL( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return 0;
}

Float_t Cppyy::CallF( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return 0.f;
}

Double_t Cppyy::CallD( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return 0.;
}

void* Cppyy::CallR( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return NULL;
}

Char_t* Cppyy::CallS( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */ ) {
   return (Char_t*)NULL;
}

Cppyy::TCppObject_t Cppyy::CallConstructor( TCppMethod_t /* method */,
      TCppType_t /* klass */, int /* nargs */, void* /* args */ ) {
   return (TCppObject_t)0;
}

Cppyy::TCppObject_t Cppyy::CallO( TCppMethod_t /* method */,
      TCppObject_t /* self */, int /* nargs */, void* /* args */, TCppType_t /* result_type */ ) {
   return (TCppObject_t)0;
}

Cppyy::TCppMethPtrGetter_t Cppyy::GetMethPtrGetter(
      TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return (TCppMethPtrGetter_t)0;
}


// handling of function argument buffer --------------------------------------
void* Cppyy::AllocateFunctionArgs( size_t /* nargs */ ) {
   return NULL;
}

void Cppyy::DeallocateFunctionArgs( void* /* args */ ) {
   /* empty */
}

size_t Cppyy::GetFunctionArgSizeof() {
   return (size_t)0;
}

size_t Cppyy::GetFunctionArgTypeoffset() {
   return (size_t)0;
}


// scope reflection information ----------------------------------------------
Bool_t Cppyy::IsNamespace( TCppScope_t handle ) {
// Test if this scope represents a namespace.
   TClassRef& cr = type_from_handle( handle );
   if ( cr.GetClass() )
      return cr->Property() & kIsNamespace;
   return kFALSE;
}   

Bool_t Cppyy::IsEnum( const std::string& /* type_name */ ) {
   return kFALSE;
}
    
    
// class reflection information ----------------------------------------------
std::string Cppyy::GetFinalName( TCppType_t klass ) {
   // TODO: either this or GetScopedFinalName is wrong
   TClassRef& cr = type_from_handle( klass );
   return ResolveName( cr->GetName() );
}

std::string Cppyy::GetScopedFinalName( TCppType_t klass ) {
   // TODO: either this or GetFinalName is wrong
   TClassRef& cr = type_from_handle( klass );
   return ResolveName( cr->GetName() );
}   

Bool_t Cppyy::HasComplexHierarchy( TCppType_t /* handle */ ) {
// Always TRUE for now (pre-empts certain optimizations).
  return kTRUE;
}

Cppyy::TCppIndex_t Cppyy::GetNumBases( TCppType_t klass ) {
// Get the total number of base classes that this class has.
   TClassRef& cr = type_from_handle( klass );
   if ( cr.GetClass() && cr->GetListOfBases() != 0 )
      return cr->GetListOfBases()->GetSize();
   return 0;
}

std::string Cppyy::GetBaseName( TCppType_t klass, TCppIndex_t ibase ) {
   TClassRef& cr = type_from_handle( klass );
   return ((TBaseClass*)cr->GetListOfBases()->At( ibase ))->GetName();
}

Bool_t Cppyy::IsSubtype( TCppType_t /* derived */, TCppType_t /* base */ ) {
   return kFALSE;
}


// method/function reflection information ------------------------------------
Cppyy::TCppIndex_t Cppyy::GetNumMethods( TCppScope_t handle ) {
   TClassRef& cr = type_from_handle(handle);
   if ( cr.GetClass() && cr->GetListOfMethods() )
      return (TCppIndex_t)cr->GetListOfMethods()->GetSize();
   else if ( handle == (TCppScope_t)GLOBAL_HANDLE ) {
      // TODO: make sure the following is done lazily instead
      std::cerr << " GetNumMethods on global scope must be made lazy " << std::endl;
      if (g_globalfuncs.empty()) {
         TCollection* funcs = gROOT->GetListOfGlobalFunctions( kTRUE );
         g_globalfuncs.reserve(funcs->GetSize());

         TIter ifunc(funcs);

         TFunction* func = 0;
         while ((func = (TFunction*)ifunc.Next()))
            g_globalfuncs.push_back(*func);
      }
      return (TCppIndex_t)g_globalfuncs.size();
   }
   return (TCppIndex_t)0;
}

Cppyy::TCppIndex_t Cppyy::GetMethodIndexAt( TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return (TCppIndex_t)0;
}

Cppyy::TCppIndex_t* Cppyy::GetMethodIndicesFromName(
      TCppScope_t /* scope */, const std::string& /* name */ ) {
   return (TCppIndex_t*)NULL;
}

std::string Cppyy::GetMethodName( TCppScope_t handle, TCppIndex_t imeth ) {
   TClassRef& cr = type_from_handle( handle );
   return ((TMethod*)cr->GetListOfMethods()->At( imeth ))->GetName();
}

std::string Cppyy::GetMethodResultType( TCppScope_t handle, TCppIndex_t imeth ) {
   TClassRef& cr = type_from_handle( handle );
   if ( cr.GetClass() && IsConstructor( handle, imeth ) )
       return "constructor";
   TFunction* f = type_get_method( handle, imeth );
   return f->GetReturnTypeName();
}

Cppyy::TCppIndex_t Cppyy::GetMethodNumArgs( TCppScope_t handle, TCppIndex_t imeth ) {
   TFunction* f = type_get_method( handle, imeth );
   return (TCppIndex_t)f->GetNargs();
}

Cppyy::TCppIndex_t Cppyy::GetMethodReqArgs( TCppScope_t handle, TCppIndex_t imeth ) {
   TFunction* f = type_get_method( handle, imeth );
   return (TCppIndex_t)(f->GetNargs() - f->GetNargsOpt());
}

std::string Cppyy::GetMethodArgType(
      TCppScope_t /* scope */, TCppIndex_t /* imeth */, int /* iarg */ ) {
   return "<unknown>";
}

std::string Cppyy::GetMethodArgDefault(
      TCppScope_t /* scope */, TCppIndex_t /* imeth */, int /* iarg */ ) {
   return "<unknown>";
}

std::string Cppyy::GetMethodSignature( TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return "<unknown>";
}

Bool_t Cppyy::IsMethodTemplate( TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return kFALSE;
}

Cppyy::TCppIndex_t Cppyy::GetMethodNumTemplateArgs(
      TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return (TCppIndex_t)0;
}

std::string Cppyy::GetMethodTemplateArgName(
      TCppScope_t /* scope */, TCppIndex_t /* imeth */, TCppIndex_t /* iarg */ ) {
   return "<unknown>";
}

Cppyy::TCppMethod_t Cppyy::GetMethod( TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return (Cppyy::TCppMethod_t)0;
}

Cppyy::TCppIndex_t Cppyy::GetGlobalOperator(
      TCppScope_t /* scope */, TCppType_t /* lc */, TCppType_t /* rc */, const std::string& /* op */ ) {
   return (TCppIndex_t)0;
}

// method properties ---------------------------------------------------------
Bool_t Cppyy::IsConstructor( TCppType_t klass, TCppIndex_t imeth ) {
   TClassRef& cr = type_from_handle( klass );
   if ( cr->Property() & kIsNamespace )
      return kFALSE;
   TMethod* m = (TMethod*)cr->GetListOfMethods()->At( imeth );
   return m->ExtraProperty() & kIsConstructor;
}

Bool_t Cppyy::IsPublicMethod( TCppType_t klass, TCppIndex_t imeth ) {
   TClassRef& cr = type_from_handle( klass );
   if ( cr->Property() & kIsNamespace )
      return kTRUE;
   TMethod* m = (TMethod*)cr->GetListOfMethods()->At( imeth );
   return m->Property() & kIsPublic;
}

Bool_t Cppyy::IsStaticMethod( TCppType_t klass, TCppIndex_t imeth ) {
   TClassRef& cr = type_from_handle( klass );
   if ( cr->Property() & kIsNamespace )
      return kTRUE;
   TMethod* m = (TMethod*)cr->GetListOfMethods()->At( imeth );
   return m->Property() & kIsStatic;
}

// data member reflection information ----------------------------------------
Cppyy::TCppIndex_t Cppyy::GetNumDatamembers( TCppScope_t handle ) {
   TClassRef& cr = type_from_handle( handle );
   if ( cr.GetClass() && cr->GetListOfDataMembers() )
      return cr->GetListOfDataMembers()->GetSize();
   else if ( handle == (TCppScope_t)GLOBAL_HANDLE ) {
      std::cerr << "   GLOBAL DATA SHOULD BE RETRIEVED LAZILY! " << std::endl;
      TCollection* vars = gROOT->GetListOfGlobals( kTRUE ); 
      if ( g_globalvars.size() != (GlobalVars_t::size_type)vars->GetSize() ) {
         g_globalvars.clear();
         g_globalvars.reserve(vars->GetSize());

         TIter ivar(vars);

         TGlobal* var = 0;
         while ( (var = (TGlobal*)ivar.Next()) )
            g_globalvars.push_back( *var );
      }
      return (TCppIndex_t)g_globalvars.size();
   }
   return (TCppIndex_t)0;
}

std::string Cppyy::GetDatamemberName( TCppScope_t handle, TCppIndex_t idata ) {
   TClassRef& cr = type_from_handle( handle );
   if (cr.GetClass()) {
      TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At( idata );
      return m->GetName();
   }
   assert( handle == (TCppScope_t)GLOBAL_HANDLE );
   TGlobal& gbl = g_globalvars[ idata ];
   return gbl.GetName();
}

std::string Cppyy::GetDatamemberType( TCppScope_t handle, TCppIndex_t idata ) {
   TClassRef& cr = type_from_handle(handle);
   if ( cr.GetClass() )  {
      TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At( idata );
      std::string fullType = m->GetFullTypeName();
      if ( (int)m->GetArrayDim() > 1 || (!m->IsBasic() && m->IsaPointer()) )
         fullType.append( "*" );
      else if ( (int)m->GetArrayDim() == 1 ) {
         std::ostringstream s;
         s << '[' << m->GetMaxIndex( 0 ) << ']' << std::ends;
         fullType.append( s.str() );
      }
      return fullType;
   }
   assert( handle == (TCppScope_t)GLOBAL_HANDLE);
   TGlobal& gbl = g_globalvars[ idata ];
   return gbl.GetFullTypeName();
}

ptrdiff_t Cppyy::GetDatamemberOffset( TCppScope_t handle, TCppIndex_t idata ) {
   TClassRef& cr = type_from_handle( handle );
   if (cr.GetClass()) {
      TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At( idata );
      return (ptrdiff_t)m->GetOffsetCint();      // yes, CINT ...
   }
   assert( handle == (TCppScope_t)GLOBAL_HANDLE );
   TGlobal& gbl = g_globalvars[ idata ];
   return (ptrdiff_t)gbl.GetAddress();
}

Cppyy::TCppIndex_t Cppyy::GetDatamemberIndex( TCppScope_t /* scope */, const std::string& /* name */ ) {
   return (TCppIndex_t)0;
}


// data member properties ----------------------------------------------------
Bool_t Cppyy::IsPublicData( TCppScope_t handle, TCppIndex_t idata ) {
   TClassRef& cr = type_from_handle( handle );
   if ( cr->Property() & kIsNamespace )
      return kTRUE;
   TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At( idata );
   return m->Property() & kIsPublic;
}

Bool_t Cppyy::IsStaticData( TCppScope_t handle, TCppIndex_t idata  ) {
   TClassRef& cr = type_from_handle( handle );
   if ( cr->Property() & kIsNamespace )
      return kTRUE;
   TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At( idata );
   return m->Property() & kIsStatic;
}

Bool_t Cppyy::IsEnumData( TCppScope_t handle, TCppIndex_t idata ) {
   TClassRef& cr = type_from_handle( handle );
   TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At( idata );
   return m->Property() & kIsEnum;
}

