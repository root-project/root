/*
  File: roottest/python/basic/Overloads.C
  Author: WLavrijsen@lbl.gov
  Created: 04/15/05
  Last: 03/08/06
*/

#include "TError.h"
#include <string>

class MyA {
public:
   MyA() { i1 = 42; i2 = -1; }
   int i1, i2;
};

class MyB {
public:
   MyB() { i1 = -2; i2 = 13; }
   int i1, i2;
};

class MyC {
public:
   MyC() {}
   int GetInt( MyA* a )   { return a->i1; }
   int GetInt( short* p ) { return *p; }
   int GetInt( MyB* b )   { return b->i2; }
   int GetInt( int* p )   { return *p; }
};

class MyD {
public:
   MyD() {}
//   int GetInt( void* p ) { return *(int*)p; }
   int GetInt( int* p )   { return *p; }
   int GetInt( MyB* b )   { return b->i2; }
   int GetInt( short* p ) { return *p; }
   int GetInt( MyA* a )   { return a->i1; }
};


class AA {};
class BB;
class CC {};
class DD;

class MyOverloads {
public:
   MyOverloads() {}
   std::string call( const AA& ) { return "AA"; }
   std::string call( const BB&, void* n = 0 ) { n = 0; return "BB"; }
   std::string call( const CC& ) { return "CC"; }
   std::string call( const DD& ) { return "DD"; }

   std::string call( double ) { return "double"; }
   std::string call( int ) { return "int"; }
   std::string call1( int ) { return "int"; }
   std::string call1( double ) { return "double"; }
};
