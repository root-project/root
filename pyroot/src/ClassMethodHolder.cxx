// Author: Wim Lavrijsen, Aug 2004

// Bindings
#include "PyROOT.h"
#include "ClassMethodHolder.h"
#include "ObjectHolder.h"
#include "Utility.h"

// ROOT
#include "TClass.h"
#include "TMethod.h"
#include "TMethodCall.h"


//- constructor -----------------------------------------------------------------
PyROOT::ClassMethodHolder::ClassMethodHolder( TClass* cls, TMethod* tm ) :
      MethodHolder( cls, tm ) {
}


//- public members --------------------------------------------------------------
PyObject* PyROOT::ClassMethodHolder::operator()( PyObject* aTuple, PyObject* /* aDict */ ) {
// setup as necessary
   if ( ! initialize() )
      return 0;                              // important: 0, not PyNone

// translate the arguments
   if ( ! setMethodArgs( aTuple, 0 ) )
      return 0;                              // important: 0, not PyNone

// execute function
   return callMethod( 0 );
}
