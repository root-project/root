#include <iostream>

enum myEnum { cow = 0, bird, fish };

class Monkey {
public:
   myEnum       testEnum1( myEnum e ) { return e; }
   unsigned int testEnum2( unsigned int e ) { return e; }
   int          testEnum3( int e ) { return e; }
};
