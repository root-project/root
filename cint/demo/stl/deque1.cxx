/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// still have problem running this program
#if 1
#include <iostream>
#include <string>
#include <cassert>
#include <deque>
#include <algorithm>
using namespace std;
#else
#include <iostream.h>
#include <string.h>
#include <assert.h>
#include <deque.h>
#include <algo.h>
#endif

int main()
{
  cout << "This program still have problem running on cint" << endl;
  cout << "Demonstrating generic find algorithm with "
    << "a deque." << endl;
  char * s = "C++ is a better C";
  int len = strlen(s);

  cout << "instantiate deque" << endl;

  // deque1 initialization
  deque<char> deque1(&s[0] , &s[len]);

  // 
  deque<char>::iterator where = find(deque1.begin(),deque1.end(), 'e');
  assert(*where == 'e' && *(++where)=='t');
  cout << *(--where) << *(++where) << endl;
}

