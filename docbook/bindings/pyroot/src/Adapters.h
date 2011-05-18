#ifndef PYROOT_ADAPTERS_H
#define PYROOT_ADAPTERS_H

// ROOT
#include "TClassRef.h"
class TBaseClass;
class TDictionary;
class TMethod;
class TFunction;
class TDataMember;
class TMethodArg;

#ifdef PYROOT_USE_REFLEX
#include "Reflex/Kernel.h"
#else
namespace ROOT {
   namespace Reflex {

   /** enum for printing names */
      enum ENTITY_HANDLING {
         FINAL     = (1<<0),
         QUALIFIED = (1<<1),
         SCOPED    = (1<<2),
         F         = (1<<4),
         Q         = (1<<5),
         S         = (1<<6)
      };

   } // namespace Reflex
} // namespace ROOT
#endif

// Standard
#include <string>


namespace PyROOT {

class TScopeAdapter;


class TReturnTypeAdapter {
public:
   TReturnTypeAdapter( const std::string& name ) : fName( name ) {}

   std::string Name( unsigned int mod = 0 ) const;

private:
   std::string fName;
};


class TMemberAdapter {
public:
   TMemberAdapter( TMethod* meth );
   operator TMethod*() const;

   TMemberAdapter( TFunction* func );
   operator TFunction*() const;

   TMemberAdapter( TDataMember* mb );
   operator TDataMember*() const;

   TMemberAdapter( TMethodArg* ma );
   operator TMethodArg*() const;

   operator bool() const { return fMember != 0; }

public:
   std::string Name( unsigned int mod = 0 ) const;

   Bool_t IsEnum() const;
   Bool_t IsPublic() const;
   Bool_t IsStatic() const;

   size_t FunctionParameterSize( bool required = false ) const;
   TMemberAdapter FunctionParameterAt( size_t nth ) const;
   std::string FunctionParameterNameAt( size_t nth ) const;
   std::string FunctionParameterDefaultAt( size_t nth ) const;

   TReturnTypeAdapter ReturnType() const;
   TScopeAdapter DeclaringScope() const;

   TMemberAdapter TypeOf() const { return *this; }

private:
   TDictionary* fMember;
};


class TBaseAdapter {
public:
   TBaseAdapter( TBaseClass* base ) : fBase( base ) {}
   operator bool() const { return fBase != 0; }

public:
   std::string Name() const;

private:
   TBaseClass* fBase;
};


class TScopeAdapter {
public:
   TScopeAdapter( TClass* klass );
   TScopeAdapter( const std::string& name );
   TScopeAdapter( const TMemberAdapter& );
   operator TClass*() const { return fClass.GetClass(); }
   operator bool() const;

public:
   static TScopeAdapter ByName( const std::string & name );

public:
   std::string Name( unsigned int mod = 0 ) const;

   TBaseAdapter BaseAt( size_t nth ) const;
   size_t BaseSize() const;

   TMemberAdapter FunctionMemberAt( size_t nth ) const;
   size_t FunctionMemberSize() const;

   TMemberAdapter DataMemberAt( size_t nth ) const;
   size_t DataMemberSize() const;

public:
   Bool_t IsComplete() const;

   Bool_t IsClass() const;
   Bool_t IsStruct() const;
   Bool_t IsNamespace() const;

// note: in Reflex, part of Type, not Scope
   Bool_t IsAbstract() const;

   void* Id() const { return fClass.GetClass(); }

private:
   TClassRef fClass;
   std::string fName;
};

} // namespace PyROOT

#endif // !PYROOT_ADAPTERS_H
