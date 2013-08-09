#include "TPython.h"

void PyClassTest1() {
// python classes live in "MyPyClass.py", which must have been loaded in a
// separate line or macro, or this one won't compile

// test generation and usability of python class
   MyPyClass m;
   printf( "string (aap) : %s\n", (const char*)(*(TPyReturn*)m.gime( "aap" ))  );
   printf( "string (noot): %s\n", (const char*) ((TPyReturn&)m.gime( "noot" )) );
   printf( "string (mies): %s\n", (const char*)              m.gime( "mies" )  );
   printf( "string (zus) : %s\n",       (char*)              m.gime( "zus" )   );

   printf( "double (0.123): %.3f\n",   (double)(*(TPyReturn*)m.gime( 0.123 ))  );
   printf( "double (0.456): %.3f\n",   (double)  ((TPyReturn)m.gime( 0.456 ))  );
   printf( "double (0.789): %.3f\n",   (double)              m.gime( 0.789 )   );
}
