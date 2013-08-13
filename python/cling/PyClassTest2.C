#include "TPython.h"

void PyClassTest2() {
// python classes live in "MyPyClass.py" and in "MyOtherPyClass.py", which
// must have been loaded in a separate line or macro, or this one won't compile

// did generation of the second class interfere with the old class?
   MyPyClass m;
   printf( "string (jet) : %s\n",       (char*)              m.gime( "zus" )   );

   MyOtherPyClass o;
   o.hop();
   o.duck();

// finally, ref counting test
   TPython::Exec( "print \'instance count:\', MyOtherPyClass.count" );
   TPython::Exec( "op = MyOtherPyClass()" );
   TPython::Exec( "print \'instance count:\', MyOtherPyClass.count" );
   TPython::Eval( "op" );
   TPython::Exec( "del op" );
   TPython::Exec( "print \'instance count:\', MyOtherPyClass.count" );
}
