// Bindings
#include "PyROOT.h"
#include "Adapters.h"

// ROOT
#include "TBaseClass.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TDataType.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TFunction.h"
#include "TMethodArg.h"
#include "TList.h"
#include "TError.h"

// CINT
#include "Api.h"


//= TReturnTypeAdapter =======================================================
std::string PyROOT::TReturnTypeAdapter::Name( unsigned int mod ) const
{
// get the name of the return type that is being adapted
   std::string name = fName;

   if ( ! ( mod & ( ROOT::Reflex::QUALIFIED | ROOT::Reflex::Q ) ) )
      name = TClassEdit::CleanType( fName.c_str(), 1 );

   if ( mod & ( ROOT::Reflex::FINAL | ROOT::Reflex::F ) )
      return TClassEdit::ResolveTypedef( name.c_str(), true );

   return name;
}


//= TMemberAdapter ===========================================================
PyROOT::TMemberAdapter::TMemberAdapter( TMethod* meth ) : fMember( meth )
{
   /* empty */
}

//____________________________________________________________________________
PyROOT::TMemberAdapter::operator TMethod*() const
{
// cast the adapter to a TMethod* being adapted, returns 0 on failure
   return dynamic_cast< TMethod* >( const_cast< TDictionary* >( fMember ) );
}

//____________________________________________________________________________
PyROOT::TMemberAdapter::TMemberAdapter( TFunction* func ) : fMember( func )
{
   /* empty */
}

//____________________________________________________________________________
PyROOT::TMemberAdapter::operator TFunction*() const
{
// cast the adapter to a TFunction* being adapted, returns 0 on failure
   return dynamic_cast< TFunction* >( const_cast< TDictionary* >( fMember ) );
}

//____________________________________________________________________________
PyROOT::TMemberAdapter::TMemberAdapter( TDataMember* mb ) : fMember( mb )
{
   /* empty */
}

//____________________________________________________________________________
PyROOT::TMemberAdapter::operator TDataMember*() const
{
// cast the adapter to a TDataMember* being adapted, returns 0 on failure
   return dynamic_cast< TDataMember* >( const_cast< TDictionary* >( fMember ) );
}

//____________________________________________________________________________
PyROOT::TMemberAdapter::TMemberAdapter( TMethodArg* ma ) : fMember( ma )
{
   /* empty */
}

//____________________________________________________________________________
PyROOT::TMemberAdapter::operator TMethodArg*() const
{
// cast the adapter to a TMethodArg* being adapted, returns 0 on failure
   return dynamic_cast< TMethodArg* >( const_cast< TDictionary* >( fMember ) );
}

//____________________________________________________________________________
std::string PyROOT::TMemberAdapter::Name( unsigned int mod ) const
{
// Return name of the type described by fMember
   TMethodArg* arg = (TMethodArg*)*this;

   if ( arg ) {

      std::string name = arg->GetTypeName();
      if ( mod & ( ROOT::Reflex::QUALIFIED | ROOT::Reflex::Q ) )
         name = arg->GetFullTypeName();

      if ( mod & ( ROOT::Reflex::FINAL | ROOT::Reflex::F ) )
         return TClassEdit::ResolveTypedef( name.c_str(), true );

      return name;

   } else if ( mod & ( ROOT::Reflex::FINAL | ROOT::Reflex::F ) )
      return TClassEdit::ResolveTypedef( fMember->GetName(), true );

   return fMember->GetName();
}

//____________________________________________________________________________
Bool_t PyROOT::TMemberAdapter::IsEnum() const
{
// test if the adapted member is of an enum type
   return fMember->Property() & kIsEnum;
}

//____________________________________________________________________________
Bool_t PyROOT::TMemberAdapter::IsPublic() const
{
// test if the adapted member represents an public (data) member
   return fMember->Property() & kIsPublic;
}

//____________________________________________________________________________
Bool_t PyROOT::TMemberAdapter::IsStatic() const
{
// test if the adapted member represents a class (data) member
   return fMember->Property() & G__BIT_ISSTATIC;
}

//____________________________________________________________________________
size_t PyROOT::TMemberAdapter::FunctionParameterSize( bool required ) const
{
// get the total number of parameters that the adapted function/method takes
   TFunction* func = (TFunction*)fMember;
   if ( ! func )
      return 0;

   if ( required == true )
      return func->GetNargs() - func->GetNargsOpt();

   return func->GetNargs();
}

//____________________________________________________________________________
PyROOT::TMemberAdapter PyROOT::TMemberAdapter::FunctionParameterAt( size_t nth ) const
{
// get the type info of the function parameter at position nth
   return (TMethodArg*)((TFunction*)fMember)->GetListOfMethodArgs()->At( nth );
}

//____________________________________________________________________________
std::string PyROOT::TMemberAdapter::FunctionParameterNameAt( size_t nth ) const
{
// get the formal name, if available, of the function parameter at position nth
   const char* name =
      ((TMethodArg*)((TFunction*)fMember)->GetListOfMethodArgs()->At( nth ))->GetName();

   if ( name )
      return name;
   return "";
}

//____________________________________________________________________________
std::string PyROOT::TMemberAdapter::FunctionParameterDefaultAt( size_t nth ) const
{
// get the default value, if available, of the function parameter at position nth
   TMethodArg* arg = (TMethodArg*)((TFunction*)fMember)->GetListOfMethodArgs()->At( nth );
   const char* def = arg->GetDefault();

   if ( ! def )
      return "";

// special case for strings: "some value" -> ""some value"
   if ( strstr( TClassEdit::ResolveTypedef( arg->GetTypeName(), true ).c_str(), "char*" ) ) {
      std::string sdef = "\"";
      sdef += def;
      sdef += "\"";
      return sdef;
   }

   return def;
}

//____________________________________________________________________________
PyROOT::TReturnTypeAdapter PyROOT::TMemberAdapter::ReturnType() const
{
// get the return type of the wrapped function/method
   return TReturnTypeAdapter( ((TFunction*)fMember)->GetReturnTypeName() );
}

//____________________________________________________________________________
PyROOT::TScopeAdapter PyROOT::TMemberAdapter::DeclaringScope() const
{
// get the declaring scope (class) of the wrapped function/method
   TMethod* method = (TMethod*)*this;
   if ( method )
      return method->GetClass();

// happens for free-standing functions (i.e. global scope)
   return std::string( "" );
}


//= TBaseAdapter =============================================================
std::string PyROOT::TBaseAdapter::Name() const
{
// get the name of the base class that is being adapted
   return fBase->GetName();
}


//= TScopeAdapter ============================================================
PyROOT::TScopeAdapter::TScopeAdapter( TClass* klass ) : fClass( klass )
{
// wrap a class (scope)
   if ( fClass.GetClass() != 0 )
      fName = fClass->GetName();
}

//____________________________________________________________________________
PyROOT::TScopeAdapter::TScopeAdapter( const std::string& name ) :
   fClass( name.c_str() ), fName( name )
{
   /* empty */
}

PyROOT::TScopeAdapter::TScopeAdapter( const TMemberAdapter& mb ) :
      fClass( mb.Name( ROOT::Reflex::SCOPED ).c_str() ),
      fName( mb.Name( ROOT::Reflex::Q | ROOT::Reflex::S ) )
{
   /* empty */
}

//____________________________________________________________________________
PyROOT::TScopeAdapter PyROOT::TScopeAdapter::ByName( const std::string & name )
{
// lookup a scope (class) by name
   return TClass::GetClass( name.c_str() );
}

//____________________________________________________________________________
std::string PyROOT::TScopeAdapter::Name( unsigned int mod ) const
{
// Return name of type described by fClass
   if ( ! fClass.GetClass() || ! fClass->Property() ) {
   // fundamental types have no class, and unknown classes have no property
      std::string name = fName;

      if ( ! ( mod & ( ROOT::Reflex::QUALIFIED | ROOT::Reflex::Q ) ) )
         name = TClassEdit::CleanType( fName.c_str(), 1 );

      if ( mod & ( ROOT::Reflex::FINAL | ROOT::Reflex::F ) )
         return TClassEdit::ResolveTypedef( name.c_str(), true );

      return name;
   }

   if ( mod & ( ROOT::Reflex::FINAL | ROOT::Reflex::F ) ) {
      G__ClassInfo* clInfo = (G__ClassInfo*)fClass->GetClassInfo();
      if ( mod & ( ROOT::Reflex::SCOPED | ROOT::Reflex::S ) )
         return clInfo ? clInfo->Fullname() : fClass->GetName();

   // unscoped name ...
      std::string actual = clInfo ? clInfo->Name() : fClass->GetName();

   // in case of missing dictionaries, the scope won't have been stripped
      if ( ! ( clInfo && clInfo->IsValid() ) ) {
         std::string::size_type pos = actual.substr( 0, actual.find( '<' ) ).rfind( "::" );
         if ( pos != std::string::npos ) {
         // this is somewhat of a gamble, but the alternative is a guaranteed crash
            actual = actual.substr( pos + 2, std::string::npos );
         }
      }

      return actual;

   } else if ( ! ( mod & ( ROOT::Reflex::SCOPED | ROOT::Reflex::S ) ) ) {
      G__ClassInfo* clInfo = (G__ClassInfo*)fClass->GetClassInfo();
      return clInfo ? clInfo->Name() : fClass->GetName();
   }

   return fClass->GetName();
}

//____________________________________________________________________________
size_t PyROOT::TScopeAdapter::BaseSize() const
{
// get the total number of base classes that this class has
   if ( fClass.GetClass() && fClass->GetListOfBases() != 0 )
      return fClass->GetListOfBases()->GetSize();

   return 0;
}

//____________________________________________________________________________
PyROOT::TBaseAdapter PyROOT::TScopeAdapter::BaseAt( size_t nth ) const
{
// get the nth base of this class
   return (TBaseClass*)fClass->GetListOfBases()->At( nth );
}

//____________________________________________________________________________
size_t PyROOT::TScopeAdapter::FunctionMemberSize() const
{
// get the total number of methods that this class has
   if ( fClass.GetClass() )
      return fClass->GetListOfMethods()->GetSize();

   return 0;
}

//____________________________________________________________________________
PyROOT::TMemberAdapter PyROOT::TScopeAdapter::FunctionMemberAt( size_t nth ) const
{
// get the nth method of this class
   return (TMethod*)fClass->GetListOfMethods()->At( nth );
}

//____________________________________________________________________________
size_t PyROOT::TScopeAdapter::DataMemberSize() const
{
// get the total number of data members that this class has
   if ( fClass.GetClass() )
      return fClass->GetListOfDataMembers()->GetSize();

   return 0;
}

//____________________________________________________________________________
PyROOT::TMemberAdapter PyROOT::TScopeAdapter::DataMemberAt( size_t nth ) const
{
// get the nth data member of this class
   return (TDataMember*)fClass->GetListOfDataMembers()->At( nth );
}

//____________________________________________________________________________
PyROOT::TScopeAdapter::operator bool() const
{
// check the validity of this scope (class)
   if ( fName.empty() )
      return false;

   Int_t oldEIL = gErrorIgnoreLevel;
   gErrorIgnoreLevel = 3000;
   bool b = G__TypeInfo( Name( ROOT::Reflex::Q | ROOT::Reflex::S ).c_str() ).IsValid();
   gErrorIgnoreLevel = oldEIL;
   return b;
}
      
//____________________________________________________________________________
Bool_t PyROOT::TScopeAdapter::IsComplete() const
{
// verify whether the dictionary of this class is fully available
   return G__TypeInfo( Name( ROOT::Reflex::SCOPED ).c_str() ).IsLoaded();
}

//____________________________________________________________________________
Bool_t PyROOT::TScopeAdapter::IsClass() const
{
// test if this scope represents a class
   if ( fClass.GetClass() ) {
   // some inverted logic: we don't have a TClass, but a builtin will be recognized, so
   // if it is NOT a builtin, it is a class or struct (but may be missing dictionary)
      return (fClass->Property() & kIsClass) || ! (fClass->Property() & kIsFundamental);
   }

// no class can mean either is no class (i.e. builtin), or no dict but coming in
// through PyCintex/Reflex ... as a workaround, use TDataTypes that has a full
// enumeration of builtin types
   return TDataType( Name( ROOT::Reflex::F | ROOT::Reflex::S ).c_str() ).GetType() == kOther_t;
}

//____________________________________________________________________________
Bool_t PyROOT::TScopeAdapter::IsStruct() const
{
// test if this scope represents a struct
   if ( fClass.GetClass() ) {
   // same logic as for IsClass() above ...
      return (fClass->Property() & kIsStruct) || ! (fClass->Property() & kIsFundamental);
   }

// same logic as for IsClass() above ...
   return TDataType( Name( ROOT::Reflex::F | ROOT::Reflex::S ).c_str() ).GetType() == kOther_t;
}

//____________________________________________________________________________
Bool_t PyROOT::TScopeAdapter::IsNamespace() const
{
// test if this scope represents a namespace
   if ( fClass.GetClass() )
      return fClass->Property() & G__BIT_ISNAMESPACE;

   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::TScopeAdapter::IsAbstract() const
{
// test if this scope represents an abstract class
   if ( fClass.GetClass() )
      return fClass->Property() & kIsAbstract;   // assume set only for classes

   return kFALSE;
}
