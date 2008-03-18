// Author: Wim Lavrijsen   March 2008

// Bindings
#include "PyROOT.h"
#include "TPySelector.h"
#include "TPyReturn.h"
#include "ObjectProxy.h"
#include "RootWrapper.h"

//- ROOT
#include "TPython.h"
#include "TString.h"


//______________________________________________________________________________
//                      Python equivalent PROOF base class
//                      ==================================
//


//- data ---------------------------------------------------------------------
ClassImp(TPySelector)


//- private helpers ----------------------------------------------------------
void TPySelector::SetupPySelf()
{
   if ( fPySelf && fPySelf != Py_None )
      return;                      // already created ...

   TString pyfile = TString::Format( "execfile( \'%s\' )", (const char*)GetOption() );

// use TPython to ensure that the interpreter is initialized
   TPython::Exec( (const char*)pyfile );
   if ( PyErr_Occurred() ) {
      Abort( 0 );
      return;
   }

// call custom function (TODO: scan file for TSelector derived class)
   PyObject* self = (PyObject*)TPython::Eval( "GetSelector()" );
   if ( ! self || ! PyROOT::ObjectProxy_Check( self ) ) {
      if ( ! PyErr_Occurred() )
         PyErr_SetString( PyExc_RuntimeError, "could not create python selector" );
      Py_XDECREF( self );
      Abort( 0 );
      return;
   }

   Py_INCREF( self );
   Py_DECREF( fPySelf );
   fPySelf = self;

// inject ourselves into the base of self
   ((PyROOT::ObjectProxy*)fPySelf)->fObject = this;
}

//____________________________________________________________________________
void TPySelector::CallSelf( const char* method )
{
// Forward <method> to python.
   if ( ! fPySelf || fPySelf == Py_None )
      return;

   PyObject* result = PyObject_CallMethod( fPySelf, (char*)method, (char*)"" );
   if ( ! result )
      Abort( 0 );

   Py_XDECREF( result );
}


//- constructors/destructor --------------------------------------------------
TPySelector::TPySelector( TTree*, PyObject* self ) : fPySelf( 0 )
{
// Construct a TSelector derived with <self> as the underlying, which is
// generally 0 to start out with in the current PROOF framework.
   if ( self ) {
      Py_INCREF( self );
      fPySelf = self;
   } else {
      Py_INCREF( Py_None );        // using None allows clearer diagnostics
      fPySelf = Py_None;
   }
}

//____________________________________________________________________________
TPySelector::~TPySelector()
{
// Destructor. Reference counting for the held python object is in effect.
   Py_DECREF( fPySelf );
}

//- public functions ---------------------------------------------------------
Int_t TPySelector::Version() const {
// Need some forwarding implementation here ...
   return 2;
}

//____________________________________________________________________________
Int_t TPySelector::GetEntry( Long64_t entry, Int_t getall )
{
// Boilerplate get entry; same as for generated code; not forwarded.
   return fChain ? fChain->GetTree()->GetEntry( entry, getall ) : 0;
}

//____________________________________________________________________________
void TPySelector::Init( TTree* tree )
{
// Initialize with the current tree to be used; not forwarded.
   if ( ! tree )
      return;

   fChain = tree;
}

//____________________________________________________________________________
Bool_t TPySelector::Notify()
{
// Need some implementation here ...
   return kTRUE;
}

//____________________________________________________________________________
void TPySelector::Begin( TTree* )
{
// First function called, and used to setup the python self; forward call.
   SetupPySelf();
   CallSelf( "Begin" );
}

//____________________________________________________________________________
void TPySelector::SlaveBegin( TTree* tree )
{
// First function called on worker node, needs to make sure python self is setup,
// then store the tree to be used, initialize client, and forward call.
   SetupPySelf();
   Init( tree );

   PyObject* result = 0;
   if ( tree ) {
      PyObject* pyobject = PyROOT::BindRootObject( (void*)tree, tree->IsA() );
      result = PyObject_CallMethod( fPySelf, (char*)"SlaveBegin", (char*)"O", pyobject );
      Py_DECREF( pyobject );
   } else {
      result = PyObject_CallMethod( fPySelf, (char*)"SlaveBegin", (char*)"O", Py_None );
   }

   if ( ! result )
      Abort( 0 );

   Py_XDECREF( result );
}

//____________________________________________________________________________
Bool_t TPySelector::Process( Long64_t entry )
{
// Actual processing; call is forwarded to python self.
   if ( ! fPySelf || fPySelf == Py_None ) {
   // would like to set a python error, but can't risk that in case of a
   // configuration problem, as it would be absorbed ...

   // simply returning kFALSE will not stop processing; need to set abort
      Abort( "no python selector instance available" );
      return kFALSE;
   }

   PyObject* result = PyObject_CallMethod( fPySelf, (char*)"Process", (char*)"L", entry );
   if ( ! result ) {
      Abort( 0 );
      return kFALSE;
   }

   Bool_t bresult = (Bool_t)PyLong_AsLong( result );
   Py_DECREF( result );
   return bresult;
}

//____________________________________________________________________________
void TPySelector::SlaveTerminate()
{
// End of client; call is forwarded to python self.
   CallSelf( "SlaveTerminate" );
}

//____________________________________________________________________________
void TPySelector::Terminate()
{
// End of job; call is forwarded to python self.
   CallSelf( "Terminate" );
}

//____________________________________________________________________________
void TPySelector::Abort( const char* why, EAbort what )
{
// If no 'why' given, read from python error
   if ( ! why && PyErr_Occurred() ) {
      PyObject *pytype = 0, *pyvalue = 0, *pytrace = 0;
      PyErr_Fetch( &pytype, &pyvalue, &pytrace );

   // abort is delayed (done at end of loop, message is current)
      PyObject* pystr = PyObject_Str( pyvalue );
      Abort( PyString_AS_STRING( pystr ), what );
      Py_DECREF( pystr );

      PyErr_Restore( pytype, pyvalue, pytrace );
   } else
      TSelector::Abort( why ? "" : why, what );
}
