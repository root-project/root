// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "ConstructorHolder.h"
#include "Executors.h"
#include "ObjectProxy.h"
#include "MemoryRegulator.h"
#include "Adapters.h"

// ROOT
#include "TClass.h"
#include "TMethod.h"

// CINT
#include "Api.h"

// Standard
#include <string>


//- protected members --------------------------------------------------------
template< class T, class M >
Bool_t PyROOT::TConstructorHolder< T, M >::InitExecutor_( TExecutor*& executor )
{
// pick up special case new object executor
   executor = (gExecFactories[ "__init__" ])();
   return kTRUE;
}

//- constructors -------------------------------------------------------------
template< class T, class M >
PyROOT::TConstructorHolder< T, M >::TConstructorHolder( const T& klass, const M& method ) :
      TMethodHolder< T, M >( klass, method )
{
}

//____________________________________________________________________________
namespace PyROOT {

template<>
TConstructorHolder< TScopeAdapter, TMemberAdapter >::TConstructorHolder( const TScopeAdapter& klass ) :
   TMethodHolder< TScopeAdapter, TMemberAdapter >( klass, (TFunction*)0 )
{
}

#ifdef PYROOT_USE_REFLEX
template<>
TConstructorHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >::TConstructorHolder(
      const ROOT::Reflex::Scope& klass ) :
   TMethodHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >( klass, ROOT::Reflex::Member() )
{
}
#endif

} // namespace PyROOT

//- public members -----------------------------------------------------------
template< class T, class M >
PyObject* PyROOT::TConstructorHolder< T, M >::GetDocString()
{
// GetMethod() may return an empty function if this is just a special case place holder
   std::string clName = this->GetClass().Name();
   return PyROOT_PyUnicode_FromFormat( "%s::%s%s",
      clName.c_str(), clName.c_str(), this->GetMethod() ? this->GetSignatureString().c_str() : "()" );
}

//____________________________________________________________________________
namespace PyROOT {

#ifdef PYROOT_USE_REFLEX
template<>
PyObject* TConstructorHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >::operator()(
      ObjectProxy* self, PyObject* args, PyObject* kwds, Long_t user )
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
   if ( kwds != 0 && PyDict_Size( kwds ) ) {
      PyErr_SetString( PyExc_TypeError, "keyword arguments are not yet supported" );
      return 0;
   }

// setup as necessary
   if ( ! this->Initialize() )
      return 0;                              // important: 0, not Py_None

// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = this->FilterArgs( self, args, kwds ) ) )
      return 0;

// translate the arguments
   if ( ! this->SetMethodArgs( args, user ) ) {
      Py_DECREF( args );
      return 0;
   }

// perform the call, and set address if successful
   Long_t address = (Long_t)this->Execute( 0 );
   if ( address != 0 ) {
      Py_INCREF( self );

   // TODO: Fix ownership once ObjectProxy can deal with Reflex
      self->Set( (void*)address );

   // done with self
      Py_DECREF( self );

      Py_INCREF( Py_None );
      return Py_None;                        // by definition
   }

   if ( ! PyErr_Occurred() )   // should be set, otherwise write a generic error msg
      PyErr_SetString( PyExc_TypeError, const_cast< char* >(
         ( this->GetClass().Name() + " constructor failed" ).c_str() ) );

// do not throw an exception, '0' might trigger the overload handler to choose a
// different constructor, which if all fails will throw an exception
   return 0;
}
#endif

} // namespace PyROOT

template< class T, class M >
PyObject* PyROOT::TConstructorHolder< T, M >::operator()(
      ObjectProxy* self, PyObject* args, PyObject* kwds, Long_t user )
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
   if ( kwds != 0 && PyDict_Size( kwds ) ) {
      PyErr_SetString( PyExc_TypeError, "keyword arguments are not yet supported" );
      return 0;
   }

// do not allow instantiation of abstract classes
   if ( this->GetClass().IsAbstract() ) {
      PyErr_Format( PyExc_TypeError,
         "%s is abstract and can not be instantiated", this->GetClass().Name().c_str() );
      return 0;
   }

// setup as necessary
   if ( ! this->Initialize() )
      return 0;                              // important: 0, not Py_None

// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = this->FilterArgs( self, args, kwds ) ) )
      return 0;

// translate the arguments
   if ( ! this->SetMethodArgs( args, user ) ) {
      Py_DECREF( args );
      return 0;
   }

   TClass* klass = (TClass*)this->GetClass().Id();

// perform the call (fails for loaded macro's)
   Long_t address = (Long_t)this->Execute( klass );
   if ( ! address && ( ! PyErr_Occurred() /* exception thrown */ ) ) {
   // the ctor call fails for interpreted classes, can deal with limited info, or
   // otherwise only deal with default ctor

      if ( klass->GetClassInfo() != 0 ) {
         Long_t tagnum = ((G__ClassInfo*)klass->GetClassInfo())->Tagnum();

      // data storage for an object of this class
         address = (Long_t)new char[ klass->Size() ];

      // set new globals, while saving current globals
         G__StoreEnv env;
         G__stubstoreenv( &env, (void*)address, tagnum );

      // build parsable line (gamble that the args look ok when stringyfied, which
      // works surprisingly well, as CINT appears to be clairvoyant)
         char temp[ G__ONELINE ];
         PyObject* str = 0;
         std::string fmt = "";
         if ( PyTuple_GET_SIZE( args ) == 1 ) {
            str = PyObject_Str( PyTuple_GET_ITEM( args, 0 ) );
            fmt = "{%s::%s(%s)}";
         } else {
            str = PyObject_Str( args );
            fmt = "{%s::%s%s}";
         }

         snprintf( temp, G__ONELINE, fmt.c_str(),
            klass->GetName(), klass->GetName(), PyROOT_PyUnicode_AsString( str ) );
         Py_DECREF( str );

      // execute contructor
         int known = 0;
         G__getfunction( temp, &known, G__CALLCONSTRUCTOR );

      // restore original globals
         G__stubrestoreenv( &env );

      // in case of failure, the error message will look really, really funky ...

      // CAUTION: creating an interpreted class doesn't work if it has STL type data
      // members that are initialized or otherwise touched in the ctor!

      } else if ( PyTuple_GET_SIZE( args ) == 0 ) {
      // unknown class, but can still create placeholder if size is known
         address = (Long_t)klass->New();
      }
   }

// done with filtered args
   Py_DECREF( args );

// return object if successful, lament if not
   if ( address != 0 ) {
      Py_INCREF( self );

   // note: constructors are no longer set to take ownership by default; instead that is
   // decided by the method proxy (which carries a creator flag) upon return
      self->Set( (void*)address );

   // allow lookup upon destruction on the ROOT/CINT side for TObjects
      TObject* object = (TObject*) klass->DynamicCast( TObject::Class(), (void*)address );
      if ( object )
         TMemoryRegulator::RegisterObject( self, object );

   // done with self
      Py_DECREF( self );

      Py_INCREF( Py_None );
      return Py_None;                        // by definition
   }

   if ( ! PyErr_Occurred() )   // should be set, otherwise write a generic error msg
      PyErr_SetString( PyExc_TypeError, const_cast< char* >(
         ( std::string( klass->GetName() ) + " constructor failed" ).c_str() ) );

// do not throw an exception, '0' might trigger the overload handler to choose a
// different constructor, which if all fails will throw an exception
   return 0;
}

//____________________________________________________________________________
template class PyROOT::TConstructorHolder< PyROOT::TScopeAdapter, PyROOT::TMemberAdapter >;
#ifdef PYROOT_USE_REFLEX
template class PyROOT::TConstructorHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >;
#endif
