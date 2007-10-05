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
   PyObject* Dispatch( const char* format = 0, ... );
#else
   PyObject* Dispatch( const char* format, ... );
#endif

   ClassDef( TPyDispatcher, 1 );   // Python dispatcher class

private:
   PyObject* fCallable;            //! callable object to be dispatched
};

#endif
