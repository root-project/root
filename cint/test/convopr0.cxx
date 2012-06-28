/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <cstdio>
#include <iostream>

using namespace std;

class A
{
private:
   unsigned int a;
   const char* p;
public:
   A(unsigned int ain, const char* pin)
   {
      a = ain;
      p = pin;
   }
   operator unsigned int()
   {
      return a;
   }
   operator const char*()
   {
      return p;
   }
};

int main()
{
   A x(123, "abcde");
   unsigned int a = (unsigned int) x;
   const char* p = (const char*) x;
   for (int i = 0; i < 5; ++i) {
      a = (unsigned int) x;
      p = (const char*) x;
      cout <<  p << a << endl;
      printf("<%s>\n", (const char*) x);
      cout << "a=" << (unsigned int) x << endl;
      cout << "p=" << (const char*) x << endl;
   }
   return 0;
}

