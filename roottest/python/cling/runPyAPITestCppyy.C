#include "TPython.h"
#include "TError.h"

void runPyAPITestCppyy() {

// The higher warning ignore level is to suppress warnings about
// classes already being in the class table (on Mac).
   int eil = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kError;

   TObject* o = new TObject;
   PyObject* a = TPython::CPPInstance_FromVoidPtr( o, "TObject" );
   printf( "OBC:  should be true:  %d\n", TPython::CPPInstance_Check( a ) );
   printf( "OBCE: should be false: %d\n", TPython::CPPInstance_CheckExact( a ) );

   Bool_t match = ( o == TPython::CPPInstance_AsVoidPtr( a ) );
   printf( "VPC:  should be true:  %d\n", match );
}
