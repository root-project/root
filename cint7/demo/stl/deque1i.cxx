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
  int  s[] = { 1,5,2,4,1,3,5,7,9,0};
  int len = sizeof(s)/sizeof(int);

  cout << "instantiate deque" << endl;

  // deque1 initialization
#ifdef G__GNUC
  deque<int> deque1(&s[0] , &s[len]);
#else
  deque<int> deque1;
  for(int i=0;i<len;i++) deque1.push_back(s[i]);
#endif

  // 
  deque<int>::iterator where = find(deque1.begin(),deque1.end(), 2);
  assert(*where == 2 && *(++where)==4);
  cout << *(--where) << *(++where) << endl;
}

