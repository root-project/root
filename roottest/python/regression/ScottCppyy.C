#include "TPython.h"
#include "CPyCppyy/PyException.h"
#include "DllImport.h"

extern "C" {
   struct _object;
   void PyErr_SetString( _object*, const char* );
   R__EXTERN _object* PyExc_SyntaxError;
}

void ThrowPyException() {
   PyErr_SetString( PyExc_SyntaxError, "test error message" );
   throw CPyCppyy::PyException();
}


class MyThrowingClass {
public:
   static void ThrowPyException( int ) {
      PyErr_SetString( PyExc_SyntaxError, "overloaded int test error message" );
      throw CPyCppyy::PyException();
   }

   static void ThrowPyException( double ) {
      PyErr_SetString( PyExc_SyntaxError, "overloaded double test error message" );
      throw CPyCppyy::PyException();
   }

};
