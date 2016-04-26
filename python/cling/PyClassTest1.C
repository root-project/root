#include "TPython.h"

void PyClassTest1() {
// python classes live in "MyPyClass.py", which must have been loaded in a
// separate line or macro, or this one won't compile

// test generation and usability of python class
   MyPyClass m;

   printf( "string   (aap): %s\n", (const char*) m.gime( "aap" )  );
   printf( "string  (noot): %s\n",       (char*) m.gime( "noot" ) );
   printf( "long     (123): %ld\n",       (long) m.gime( 123 )    );
   printf( "float  (0.456): %.3f\n",     (float) m.gime( 0.456 )  );
   printf( "double (0.789): %.3f\n",    (double) m.gime( 0.789 )  );

   fflush( stdout );
}
