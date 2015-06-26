// Bindings
#include "PyROOT.h"
#include "Adapters.h"
#include "Utility.h"

// ROOT
#include "TInterpreter.h"
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


//- helper -------------------------------------------------------------------
namespace PyROOT {
   inline std::string UnqualifiedTypeName( const std::string name ) {
      return TClassEdit::ShortType(
         TClassEdit::CleanType( name.c_str(), 1 ).c_str(), 5 );
   }
} // namespace PyROOT

//= TReturnTypeAdapter =======================================================
std::string PyROOT::TReturnTypeAdapter::Name( unsigned int mod ) const
{
// get the name of the return type that is being adapted
   std::string name = fName;

   if ( mod & Rflx::FINAL )
      name = Utility::ResolveTypedef( name );

   if ( ! ( mod & Rflx::QUALIFIED ) )
      name = UnqualifiedTypeName( fName );

   return name;
}


//= TMemberAdapter ===========================================================
PyROOT::TMemberAdapter::TMemberAdapter( TMethod* meth ) : fMember( meth )
{
   /* empty */
}

////////////////////////////////////////////////////////////////////////////////
/// cast the adapter to a TMethod* being adapted, returns 0 on failure

PyROOT::TMemberAdapter::operator TMethod*() const
{
   return dynamic_cast< TMethod* >( const_cast< TDictionary* >( fMember ) );
}

////////////////////////////////////////////////////////////////////////////////

PyROOT::TMemberAdapter::TMemberAdapter( TFunction* func ) : fMember( func )
{
   /* empty */
}

////////////////////////////////////////////////////////////////////////////////
/// cast the adapter to a TFunction* being adapted, returns 0 on failure

PyROOT::TMemberAdapter::operator TFunction*() const
{
   return dynamic_cast< TFunction* >( const_cast< TDictionary* >( fMember ) );
}

////////////////////////////////////////////////////////////////////////////////

PyROOT::TMemberAdapter::TMemberAdapter( TDataMember* mb ) : fMember( mb )
{
   /* empty */
}

////////////////////////////////////////////////////////////////////////////////
/// cast the adapter to a TDataMember* being adapted, returns 0 on failure

PyROOT::TMemberAdapter::operator TDataMember*() const
{
   return dynamic_cast< TDataMember* >( const_cast< TDictionary* >( fMember ) );
}

////////////////////////////////////////////////////////////////////////////////

PyROOT::TMemberAdapter::TMemberAdapter( TMethodArg* ma ) : fMember( ma )
{
   /* empty */
}

////////////////////////////////////////////////////////////////////////////////
/// cast the adapter to a TMethodArg* being adapted, returns 0 on failure

PyROOT::TMemberAdapter::operator TMethodArg*() const
{
   return dynamic_cast< TMethodArg* >( const_cast< TDictionary* >( fMember ) );
}

////////////////////////////////////////////////////////////////////////////////
/// Return name of the type described by fMember

std::string PyROOT::TMemberAdapter::Name( unsigned int mod ) const
{
   TMethodArg* arg = (TMethodArg*)*this;

   if ( arg ) {

      std::string name = arg->GetTypeNormalizedName();
      if ( mod & Rflx::FINAL )
         name = Utility::ResolveTypedef( name );

      if ( ! ( mod & Rflx::QUALIFIED ) )
         name = UnqualifiedTypeName( name );

      return name;

   } else if ( mod & Rflx::FINAL )
      return Utility::ResolveTypedef( fMember->GetName() );

   if ( fMember )
      return fMember->GetName();
   return "<unknown>";   // happens for classes w/o dictionary
}

////////////////////////////////////////////////////////////////////////////////
/// test if the adapted member is a const method

Bool_t PyROOT::TMemberAdapter::IsConstant() const
{
   return fMember->Property() & kIsConstMethod;
}

////////////////////////////////////////////////////////////////////////////////
/// test if the adapted member is a const method

Bool_t PyROOT::TMemberAdapter::IsConstructor() const
{
   return ((TFunction*)fMember) ? (((TFunction*)fMember)->ExtraProperty() & kIsConstructor) : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// test if the adapted member is of an enum type

Bool_t PyROOT::TMemberAdapter::IsEnum() const
{
   return fMember->Property() & kIsEnum;
}

////////////////////////////////////////////////////////////////////////////////
/// test if the adapted member represents an public (data) member

Bool_t PyROOT::TMemberAdapter::IsPublic() const
{
   return fMember->Property() & kIsPublic;
}

////////////////////////////////////////////////////////////////////////////////
/// test if the adapted member represents a class (data) member

Bool_t PyROOT::TMemberAdapter::IsStatic() const
{
   if ( DeclaringScope().IsNamespace() )
      return kTRUE;
   return fMember->Property() & kIsStatic;
}

////////////////////////////////////////////////////////////////////////////////
/// get the total number of parameters that the adapted function/method takes

size_t PyROOT::TMemberAdapter::FunctionParameterSize( Bool_t required ) const
{
   TFunction* func = (TFunction*)fMember;
   if ( ! func )
      return 0;

   if ( required == true )
      return func->GetNargs() - func->GetNargsOpt();

   return func->GetNargs();
}

////////////////////////////////////////////////////////////////////////////////
/// get the type info of the function parameter at position nth

PyROOT::TMemberAdapter PyROOT::TMemberAdapter::FunctionParameterAt( size_t nth ) const
{
   return (TMethodArg*)((TFunction*)fMember)->GetListOfMethodArgs()->At( nth );
}

////////////////////////////////////////////////////////////////////////////////
/// get the formal name, if available, of the function parameter at position nth

std::string PyROOT::TMemberAdapter::FunctionParameterNameAt( size_t nth ) const
{
   const char* name =
      ((TMethodArg*)((TFunction*)fMember)->GetListOfMethodArgs()->At( nth ))->GetName();

   if ( name )
      return name;
   return "";
}

////////////////////////////////////////////////////////////////////////////////
/// get the default value, if available, of the function parameter at position nth

std::string PyROOT::TMemberAdapter::FunctionParameterDefaultAt( size_t nth ) const
{
   TMethodArg* arg = (TMethodArg*)((TFunction*)fMember)->GetListOfMethodArgs()->At( nth );
   const char* def = arg->GetDefault();

   if ( ! def )
      return "";

// special case for strings: "some value" -> ""some value"
   if ( strstr( Utility::ResolveTypedef( arg->GetTypeNormalizedName() ).c_str(), "char*" ) ) {
      std::string sdef = "\"";
      sdef += def;
      sdef += "\"";
      return sdef;
   }

   return def;
}

////////////////////////////////////////////////////////////////////////////////
/// get the return type of the wrapped function/method

PyROOT::TReturnTypeAdapter PyROOT::TMemberAdapter::ReturnType() const
{
   return TReturnTypeAdapter( ((TFunction*)fMember)->GetReturnTypeNormalizedName() );
}

////////////////////////////////////////////////////////////////////////////////
/// get the declaring scope (class) of the wrapped function/method

PyROOT::TScopeAdapter PyROOT::TMemberAdapter::DeclaringScope() const
{
   TMethod* method = (TMethod*)*this;
   if ( method )
      return method->GetClass();

   TDataMember* data = (TDataMember*)*this;
   if ( data )
      return data->GetClass();

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

////////////////////////////////////////////////////////////////////////////////

PyROOT::TScopeAdapter::TScopeAdapter( const std::string& name ) :
   fClass( name.c_str() ), fName( name )
{
   /* empty */
}

PyROOT::TScopeAdapter::TScopeAdapter( const TMemberAdapter& mb ) :
      fClass( mb.Name( Rflx::SCOPED ).c_str() ),
      fName( mb.Name( Rflx::QUALIFIED | Rflx::SCOPED ) )
{
   /* empty */
}

////////////////////////////////////////////////////////////////////////////////
/// lookup a scope (class) by name

PyROOT::TScopeAdapter PyROOT::TScopeAdapter::ByName( const std::string& name, Bool_t quiet )
{
   Int_t oldEIL = gErrorIgnoreLevel;
   if ( quiet )
      gErrorIgnoreLevel = 3000;

   TClassRef klass( name.c_str() );

   gErrorIgnoreLevel = oldEIL;

   return klass.GetClass();
}

////////////////////////////////////////////////////////////////////////////////
/// Return name of type described by fClass

std::string PyROOT::TScopeAdapter::Name( unsigned int mod ) const
{
   if ( ! fClass.GetClass() || ! fClass->Property() ) {
   // fundamental types have no class, and unknown classes have no property
      std::string name = fName;

      if ( mod & Rflx::FINAL )
         name = Utility::ResolveTypedef( name );

      if ( ! ( mod & Rflx::QUALIFIED ) )
         name = UnqualifiedTypeName( fName );

      return name;
   }

   std::string name = fClass->GetName();
   if ( mod & Rflx::FINAL )
      name = Utility::ResolveTypedef( name );

   if ( ! (mod & Rflx::SCOPED) ) {
   // remove scope from the name
      Int_t tpl_open = 0;
      for ( std::string::size_type pos = name.size() - 1; 0 < pos; --pos ) {
         std::string::value_type c = name[ pos ];

      // count '<' and '>' to be able to skip template contents
         if ( c == '>' )
            ++tpl_open;
         else if ( c == '<' )
            --tpl_open;
         else if ( tpl_open == 0 && c == ':' && 0 < pos && name[ pos-1 ] == ':' ) {
         // found scope, strip name from it
            name = name.substr( pos+1, std::string::npos );
            break;
         }
      }
   }

   return name;
}

////////////////////////////////////////////////////////////////////////////////
/// get the total number of base classes that this class has

size_t PyROOT::TScopeAdapter::BaseSize() const
{
   if ( fClass.GetClass() && fClass->GetListOfBases() != 0 )
      return fClass->GetListOfBases()->GetSize();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// get the nth base of this class

PyROOT::TBaseAdapter PyROOT::TScopeAdapter::BaseAt( size_t nth ) const
{
   return (TBaseClass*)fClass->GetListOfBases()->At( nth );
}

////////////////////////////////////////////////////////////////////////////////
/// get the total number of methods that this class has

size_t PyROOT::TScopeAdapter::FunctionMemberSize() const
{
   if ( fClass.GetClass() )
      return fClass->GetListOfMethods()->GetSize();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// get the nth method of this class

PyROOT::TMemberAdapter PyROOT::TScopeAdapter::FunctionMemberAt( size_t nth ) const
{
   return (TMethod*)fClass->GetListOfMethods()->At( nth );
}

////////////////////////////////////////////////////////////////////////////////
/// get the total number of data members that this class has

size_t PyROOT::TScopeAdapter::DataMemberSize() const
{
   if ( fClass.GetClass() )
      return fClass->GetListOfDataMembers()->GetSize();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// get the nth data member of this class

PyROOT::TMemberAdapter PyROOT::TScopeAdapter::DataMemberAt( size_t nth ) const
{
   return (TDataMember*)fClass->GetListOfDataMembers()->At( nth );
}

////////////////////////////////////////////////////////////////////////////////
/// check the validity of this scope (class)

PyROOT::TScopeAdapter::operator Bool_t() const
{
   if ( fName.empty() )
      return false;

   Bool_t b = kFALSE;

   Int_t oldEIL = gErrorIgnoreLevel;
   gErrorIgnoreLevel = 3000;
   std::string scname = Name( Rflx::SCOPED );
   TClass* klass = TClass::GetClass( scname.c_str() );
   if ( klass && klass->HasInterpreterInfo() )     // works for normal case w/ dict
      b = kTRUE;
   else {      // special case for forward declared classes
      ClassInfo_t* ci = gInterpreter->ClassInfo_Factory( scname.c_str() );
      if ( ci ) {
         b = gInterpreter->ClassInfo_IsValid( ci );
         gInterpreter->ClassInfo_Delete( ci );    // we own the fresh class info
      }
   }
   gErrorIgnoreLevel = oldEIL;
   return b;
}

////////////////////////////////////////////////////////////////////////////////
/// verify whether the dictionary of this class is fully available

Bool_t PyROOT::TScopeAdapter::IsComplete() const
{
   Bool_t b = kFALSE;

   Int_t oldEIL = gErrorIgnoreLevel;
   gErrorIgnoreLevel = 3000;
   std::string scname = Name( Rflx::SCOPED );
   TClass* klass = TClass::GetClass( scname.c_str() );
   if ( klass && klass->GetClassInfo() )     // works for normal case w/ dict
      b = gInterpreter->ClassInfo_IsLoaded( klass->GetClassInfo() );
   else {      // special case for forward declared classes
      ClassInfo_t* ci = gInterpreter->ClassInfo_Factory( scname.c_str() );
      if ( ci ) {
         b = gInterpreter->ClassInfo_IsLoaded( ci );
         gInterpreter->ClassInfo_Delete( ci );    // we own the fresh class info
      }
   }
   gErrorIgnoreLevel = oldEIL;
   return b;
}

////////////////////////////////////////////////////////////////////////////////
/// test if this scope represents a class

Bool_t PyROOT::TScopeAdapter::IsClass() const
{
   if ( fClass.GetClass() ) {
   // some inverted logic: we don't have a TClass, but a builtin will be recognized, so
   // if it is NOT a builtin, it is a class or struct (but may be missing dictionary)
      return (fClass->Property() & kIsClass) || ! (fClass->Property() & kIsFundamental);
   }

// no class can mean either is no class (i.e. builtin), or no dict but coming in
// through PyCintex/Reflex ... as a workaround, use TDataTypes that has a full
// enumeration of builtin types
   return TDataType( Name( Rflx::FINAL | Rflx::SCOPED ).c_str() ).GetType() == kOther_t;
}

////////////////////////////////////////////////////////////////////////////////
/// test if this scope represents a struct

Bool_t PyROOT::TScopeAdapter::IsStruct() const
{
   if ( fClass.GetClass() ) {
   // same logic as for IsClass() above ...
      return (fClass->Property() & kIsStruct) || ! (fClass->Property() & kIsFundamental);
   }

// same logic as for IsClass() above ...
   return TDataType( Name( Rflx::FINAL | Rflx::SCOPED ).c_str() ).GetType() == kOther_t;
}

////////////////////////////////////////////////////////////////////////////////
/// test if this scope represents a namespace

Bool_t PyROOT::TScopeAdapter::IsNamespace() const
{
   if ( fClass.GetClass() )
      return fClass->Property() & kIsNamespace;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// test if this scope represents an abstract class

Bool_t PyROOT::TScopeAdapter::IsAbstract() const
{
   if ( fClass.GetClass() )
      return fClass->Property() & kIsAbstract;   // assume set only for classes

   return kFALSE;
}
