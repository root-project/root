#include "TPython.h"
#include "TError.h"

void runPyAPITest() {
// The higher warning ignore level is to suppress warnings about
// classes already being in the class table (on Mac).
   int eil = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kError;

   TObject* o = new TObject;
   PyObject* a = TPython::ObjectProxy_FromVoidPtr( o, "TObject" );
   printf( "OBC:  should be true:  %d\n", TPython::ObjectProxy_Check( a ) );
   printf( "OBCE: should be false: %d\n", TPython::ObjectProxy_CheckExact( a ) );

   Bool_t match = ( o == TPython::ObjectProxy_AsVoidPtr( a ) );
   printf( "VPC:  should be true:  %d\n", match );
}
