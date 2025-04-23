
#ifndef TEST_NESTED_VECT
#define TEST_NESTED_VECT

#include <vector>
// #include <list>

class Elem
{
public:
  Elem(int v=0) : i(v) { }
   int i = 5;
};

class Elem2
{
public:
  Elem2(int v=0) : i(v) { }
   int i = 5;
};

class ECont
{
public:
   std::vector<Elem> elems;
//    Elem2 xx;
};

#endif
