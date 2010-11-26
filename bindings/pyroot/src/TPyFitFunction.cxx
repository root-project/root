// Author: Wim Lavrijsen   March 2008

// Bindings
#include "PyROOT.h"
#include "TPyFitFunction.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "PyBufferFactory.h"

//______________________________________________________________________________
//                       Python wrapper for Fit functions
//                       ================================
//


//- data ---------------------------------------------------------------------
ClassImp(TPyMultiGenFunction)


//____________________________________________________________________________
PyObject* TPyMultiGenFunction::CallSelf( const char* method, PyObject* pyobject ) const
{
// Forward <method> to python (need to refactor this with TPySelector).
   if ( ! fPySelf || fPySelf == Py_None ) {
      Py_INCREF( Py_None );
      return Py_None;
   }

   PyObject* result = 0;

// get the named method and check for python side overload by not accepting the
// binding's methodproxy
   PyObject* pymethod = PyObject_GetAttrString( (PyObject*)fPySelf, const_cast< char* >( method ) );
   if ( ! PyROOT::MethodProxy_CheckExact( pymethod ) ) {
      if ( pyobject )
         result = PyObject_CallFunction( pymethod, const_cast< char* >( "O" ), pyobject );
      else
         result = PyObject_CallFunction( pymethod, const_cast< char* >( "" ) );
   } else {
   // silently ignore if method not overridden (note that the above can't lead
   // to a python exception, since this (TPyMultiGenFunction) class contains the
   // method it is always to be found)
      Py_INCREF( Py_None );
      result = Py_None;
   }

   Py_XDECREF( pymethod );

   return result;
}


//- constructors/destructor --------------------------------------------------
TPyMultiGenFunction::TPyMultiGenFunction( PyObject* self ) : fPySelf( 0 )
{
// Construct a TPyMultiGenFunction derived with <self> as the underlying
   if ( self ) {
   // steal reference as this is us, as seen from python
      fPySelf = self;
   } else {
      Py_INCREF( Py_None );        // using None allows clearer diagnostics
      fPySelf = Py_None;
   }
}

//____________________________________________________________________________
TPyMultiGenFunction::~TPyMultiGenFunction()
{
// Destructor. Only deref if still holding on to Py_None (circular otherwise).
   if ( fPySelf == Py_None ) {
      Py_DECREF( fPySelf );
   }
}


//- public functions ---------------------------------------------------------
TPyMultiGenFunction* TPyMultiGenFunction::Clone() const
{
   return new TPyMultiGenFunction( fPySelf );
}


//____________________________________________________________________________
unsigned int TPyMultiGenFunction::NDim() const
{
// Simply forward the call to python self.
   PyObject* pyresult = CallSelf( "NDim" );

   if ( ! pyresult )
      return 1;    // probably reasonable default

   unsigned int cppresult = (unsigned int)PyLong_AsLong( pyresult );
   Py_XDECREF( pyresult );

   return cppresult;
}


//____________________________________________________________________________
double TPyMultiGenFunction::DoEval( const double* x ) const
{
// Simply forward the call to python self.
   PyObject* xbuf = PyROOT::TPyBufferFactory::Instance()->PyBuffer_FromMemory( (Double_t*)x );
   PyObject* pyresult = CallSelf( "DoEval", xbuf );
   Py_DECREF( xbuf );

   if ( ! pyresult )
      return 1;    // probably reasonable default

   double cppresult = (double)PyFloat_AsDouble( pyresult );
   Py_XDECREF( pyresult );

   return cppresult;
}
