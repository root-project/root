#include "PyROOT.h"
#include "Cppyy.h"

// ROOT
#include "TClass.h"
#include "TClassRef.h"
#include "TClassTable.h"
#include "TClassEdit.h"
#include "TDataType.h"
#include "TFunction.h"
#include "TGlobal.h"
#include "TList.h"
#include "TROOT.h"

// Standard
#include <assert.h>
#include <map>
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
      assert( g_classrefs.size() == (ClassRefs_t::size_type)GLOBAL_HANDLE );
      g_classref_indices[ "" ] = (ClassRefs_t::size_type)GLOBAL_HANDLE;
      g_classrefs.push_back(TClassRef(""));
      // ROOT ignores std/::std, so point them to the global namespace
      g_classref_indices[ "std" ]   = (ClassRefs_t::size_type)GLOBAL_HANDLE;
      g_classref_indices[ "::std" ] = (ClassRefs_t::size_type)GLOBAL_HANDLE;
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
   assert( handle == (TCppType_t)GLOBAL_HANDLE );
   return gClassTable->Classes();
}

std::string Cppyy::GetScopeName( TCppScope_t parent, TCppIndex_t iscope ) {
// Retrieve the scope name of the scope indexed with iscope in parent.
   TClassRef& cr = type_from_handle( parent );
   if ( cr.GetClass() ) return 0;   // not supported if not at global scope
   assert( parent == (TCppType_t)GLOBAL_HANDLE );
   std::string name = gClassTable->At( iscope );
   if ( name.find("::") == std::string::npos )
       return name;
   return "";
}

std::string Cppyy::ResolveName( TCppScope_t handle ) {
   TClassRef& cr = type_from_handle( handle );
   return ResolveName( cr->GetName() );
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

Cppyy::TCppType_t Cppyy::GetActualClass( TCppType_t klass, TCppObject_t /* obj */ ) {
   return klass;
}


// memory management ---------------------------------------------------------
Cppyy::TCppObject_t Cppyy::Allocate( TCppType_t /* type */ ) {
   return (TCppObject_t)0;
}

void Cppyy::Deallocate( TCppType_t /* type */, TCppObject_t /* self */ ) {
   /* empty */
}

void Cppyy::Destruct( TCppType_t /* type */, TCppObject_t /* self */ ) {
   /* empty */
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
Bool_t Cppyy::IsNamespace( TCppScope_t /* handle */ ) {
   return kFALSE;
}   

Bool_t Cppyy::IsEnum( const std::string& /* type_name */ ) {
   return kFALSE;
}
    
    
// class reflection information ----------------------------------------------
std::string Cppyy::GetFinalName( TCppType_t /* handle */ ) {
   return "<unknown>";
}

std::string Cppyy::GetScopedFinalName( TCppType_t /* handle */ ) {
   return "<unknown>";
}   

Bool_t Cppyy::HasComplexHierarchy( TCppType_t /* handle */ ) {
   return kFALSE;
}

Cppyy::TCppIndex_t Cppyy::GetNumBases( TCppType_t /* type */ ) {
   return 0;
}

std::string Cppyy::GetBaseName( TCppType_t /* type */, TCppIndex_t /* ibase */ ) {
   return "<unknown>";
}

Bool_t Cppyy::IsSubtype( TCppType_t /* derived */, TCppType_t /* base */ ) {
   return kFALSE;
}


// method/function reflection information ------------------------------------
Cppyy::TCppIndex_t Cppyy::GetNumMethods( TCppScope_t handle ) {
   TClassRef& cr = type_from_handle(handle);
   if (cr.GetClass() && cr->GetListOfMethods())
      return (TCppIndex_t)cr->GetListOfMethods()->GetSize();
   else if (handle == (TCppScope_t)GLOBAL_HANDLE) {
      // TODO: make sure the following is done lazily instead
      std::cerr << " GetNumMethods on global scope must be made lazy " << std::endl;
      if (g_globalfuncs.empty()) {
         TCollection* funcs = gROOT->GetListOfGlobalFunctions(kTRUE);
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

std::string Cppyy::GetMethodName( TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return "<unknown>";
}

std::string Cppyy::GetMethodResultType( TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return "<unknown>";
}

Cppyy::TCppIndex_t Cppyy::GetMethodNumArgs( TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return (TCppIndex_t)0;
}

Cppyy::TCppIndex_t Cppyy::GetMethodReqArgs( TCppScope_t /* scope */, TCppIndex_t /* imeth */ ) {
   return (TCppIndex_t)0;
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
Bool_t Cppyy::IsConstructor( TCppType_t /* type */, TCppIndex_t /* imeth */ ) {
   return kFALSE;
}

Bool_t Cppyy::IsStaticMethod( TCppType_t /* type */, TCppIndex_t /* imeth */ ) {
   return kFALSE;
}

// data member reflection information ----------------------------------------
Cppyy::TCppIndex_t Cppyy::GetNumDatamembers( TCppScope_t /* scope */ ) {
   return (TCppIndex_t)0;
}

std::string Cppyy::GetDatamemberName( TCppScope_t /* scope */, TCppIndex_t /* idata */ ) {
   return "<unknown>";
}

std::string Cppyy::GetDatamemberType( TCppScope_t /* scope */, TCppIndex_t /* idata */ ) {
   return "<unknown>";
}

ptrdiff_t Cppyy::GetDatamemberOffset( TCppScope_t /* scope */, TCppIndex_t /* idata */ ) {
   return (ptrdiff_t)0;
}

Cppyy::TCppIndex_t Cppyy::GetDatamemberIndex( TCppScope_t /* scope */, const std::string& /* name */ ) {
   return (TCppIndex_t)0;
}


// data member properties ----------------------------------------------------
Bool_t Cppyy::IsPublicData( TCppType_t /* type */, TCppIndex_t /* idata */) {
   return kFALSE;
}

Bool_t Cppyy::IsStaticData( TCppType_t /* type */, TCppIndex_t /* idata */ ) {
   return kFALSE;
}
