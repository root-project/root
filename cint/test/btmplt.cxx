/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// From berlich@pc66.mppmu.mpg.de Mon May 13 11:44 JST 1996

#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

template <class tp>
void printstatement(tp stm)
{
  cout << "Argument is : " << stm << endl;
}

class test
{
public:
  test(void){printstatement("This is a very long string");}
};

template <class tp2>
class test2
{
public:
  test2(tp2);
};

template <class tp2>
test2<tp2>::test2(tp2 stm)
{
  cout << "Argument is : " << stm << endl;
}

int main(void) {
  test *Test=new test;
  delete Test;
  test2<const char*>* Test2=new test2<const char*>("This is another long string");
  delete Test2;
  return 0;
}
