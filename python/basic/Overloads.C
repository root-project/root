/*
  File: roottest/python/basic/Overloads.C
  Author: WLavrijsen@lbl.gov
  Created: 04/15/05
  Last: 12/04/10
*/

#include <string>
#include <vector>

class MyA {
public:
   MyA() { i1 = 42; i2 = -1; }
   int i1, i2;
};

namespace MyNSa {
   class MyA {
   public:
      MyA() { i1 = 88; i2 = -34; }
      int i1, i2;
   };

   class MyB {
   public:
      int f( const std::vector<int>* v ) { return (*v)[0]; }
   };
}

namespace MyNSb {
   class MyA {
   public:
      MyA() { i1 = -33; i2 = 89; }
      int i1, i2;
   };
}

class MyB {
public:
   MyB() { i1 = -2; i2 = 13; }
   int i1, i2;
};

class MyC {
public:
   MyC() {}
   int GetInt( MyA* a )   { return a->i1; }
   int GetInt( MyNSa::MyA* a ) { return a->i1; }
   int GetInt( MyNSb::MyA* a ) { return a->i1; }
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
   int GetInt( MyNSb::MyA* a ) { return a->i1; }
   int GetInt( MyNSa::MyA* a ) { return a->i1; }
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
   std::string call( const BB&, void* n = 0 ) { if (n) return "BB"; return "BB"; }
   std::string call( const CC& ) { return "CC"; }
   std::string call( const DD& ) { return "DD"; }

   std::string callUnknown( const DD& ) { return "DD"; }

   std::string call( double ) { return "double"; }
   std::string call( int ) { return "int"; }
   std::string call1( int ) { return "int"; }
   std::string call1( double ) { return "double"; }
};

class MyOverloads2 {
public:
   MyOverloads2() {}
   std::string call( const BB& ) { return "BBref"; }
   std::string call( const BB* ) { return "BBptr"; }

   std::string call( const DD*, int ) { return "DDptr"; }
   std::string call( const DD&, int ) { return "DDref"; }
};
