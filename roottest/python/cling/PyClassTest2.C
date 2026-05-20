#include "TPython.h"

void PyClassTest2() {
// python classes live in "MyPyClass.py" and in "MyOtherPyClass.py", which
// must have been loaded in a separate line or macro, or this one won't compile

// did generation of the second class interfere with the old class?
   MyPyClass m;
   printf( "string (jet) : %s\n",       (char*)              m.gime( "zus" )   );
   fflush( stdout );

   MyOtherPyClass o;
   o.hop();
   o.duck();

// finally, ref counting test
   TPython::Exec( "import sys" );
   TPython::Exec( "sys.stdout.write( \'instance count: %d\\n\' % (MyOtherPyClass.count,) )" );
   TPython::Exec( "op = MyOtherPyClass()" );
   TPython::Exec( "sys.stdout.write( \'instance count: %d\\n\' % (MyOtherPyClass.count,) )" );
   TPython::Exec( "op" );
   TPython::Exec( "del op" );
   TPython::Exec( "sys.stdout.write( \'instance count: %d\\n\' % (MyOtherPyClass.count,) )" );
}
