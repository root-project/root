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

   TString impst = TString::Format( "import %s", GetOption() );

// use TPython to ensure that the interpreter is initialized
   if ( ! TPython::Exec( (const char*)impst ) ) {
      Abort( "failed to load provided script" );  // Exec already printed the real error
      return;
   }

// get the TPySelector python class
   PyObject* tpysel = PyObject_GetAttrString(
      PyImport_AddModule( const_cast< char* >( "libPyROOT" ) ),
      const_cast< char* >( "TPySelector" ) );

// get handle to the module
   PyObject* pymod = PyImport_AddModule( const_cast< char* >( GetOption() ) );

// get the module dictionary to loop over
   PyObject* dict = PyModule_GetDict( pymod );
   Py_INCREF( dict );

// locate the TSelector derived class
   PyObject* allvalues = PyDict_Values( dict );

   PyObject* pyclass = 0;
   for ( int i = 0; i < PyList_GET_SIZE( allvalues ); ++i ) {
      PyObject* value = PyList_GET_ITEM( allvalues, i );
      Py_INCREF( value );

      if ( PyType_Check( value ) && PyObject_IsSubclass( value, tpysel ) ) {
         if ( PyObject_Compare(	value, tpysel ) ) {    // i.e., if not equal
            pyclass = value;
            break;
         }
      }

      Py_DECREF( value );
   }

   Py_DECREF( allvalues );
   Py_DECREF( dict );
   Py_DECREF( tpysel );

   if ( ! pyclass ) {
      Abort( "no TSelector derived class available in provided module" );
      return;
   }

   PyObject* args = PyTuple_New( 0 );
   PyObject* self = PyObject_Call( pyclass, args, 0 );
   Py_DECREF( args );
   Py_DECREF( pyclass );

// final check before declaring success ...
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

   PyObject* result = PyObject_CallMethod(
      fPySelf, const_cast< char* >( method ), const_cast< char* >( "" ) );
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
// Initialize with the current tree to be used; not forwarded (may be called
// multiple times, and is called from Begin() and SlaveBegin() ).
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
      result = PyObject_CallMethod( fPySelf,
         const_cast< char* >( "SlaveBegin" ), const_cast< char* >( "O" ), pyobject );
      Py_DECREF( pyobject );
   } else {
      result = PyObject_CallMethod( fPySelf,
         const_cast< char* >( "SlaveBegin" ), const_cast< char* >( "O" ), Py_None );
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

   PyObject* result = PyObject_CallMethod( fPySelf,
      const_cast< char* >( "Process" ), const_cast< char* >( "L" ), entry );
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
      TSelector::Abort( why ? why : "", what );
}
