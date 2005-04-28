/*
  File: roottest/python/basic/Overloads.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 04/15/05
  Last: 04/22/05
*/

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
