// Author: Wim Lavrijsen   Aug 2007

#ifndef ROOT_TPyDispatcher
#define ROOT_TPyDispatcher

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPyDispatcher                                                            //
//                                                                          //
// Dispatcher for CINT callbacks into python code.                          //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


// ROOT
#ifndef ROOT_TObject
#include "TObject.h"
#endif

// Python
struct _object;
typedef _object PyObject;


class TPyDispatcher : public TObject {
public:
   TPyDispatcher( PyObject* callable );
   TPyDispatcher( const TPyDispatcher& );
   TPyDispatcher& operator=( const TPyDispatcher& );
   ~TPyDispatcher();

public:
#ifndef __CINT__
   PyObject* DispatchVA( const char* format = 0, ... );
#else
   PyObject* DispatchVA( const char* format, ... );
#endif

// pre-defined dispatches, same as per TQObject::Emit(); note that
// Emit() maps exclusively to this set, so several builtin types (e.g.
// Int_t, Bool_t, Float_t, etc.) have been omitted here
   PyObject* Dispatch() { return DispatchVA( 0 ); }
   PyObject* Dispatch( const char* param ) { return DispatchVA( "s", param ); }
   PyObject* Dispatch( Double_t param )    { return DispatchVA( "d", param ); }
   PyObject* Dispatch( Long_t param )      { return DispatchVA( "l", param ); }
   PyObject* Dispatch( Long64_t param )    { return DispatchVA( "L", param ); }

   ClassDef( TPyDispatcher, 1 );   // Python dispatcher class

private:
   PyObject* fCallable;            //! callable object to be dispatched
};

#endif
