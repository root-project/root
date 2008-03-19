/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream.h>
#include <string.h>
#include <assert.h>
#include <_list.h> // problem in precompiled list class and cint
#include <algo.h>

int main()
{
  cout << "Demonstrating generic find algorithm with "
    << "a list." << endl;
  char * s = "C++ is a better C";
  int len = strlen(s);

  cout << "instantiate list" << endl;

  // list1 initialization
  list<char> list1(&s[0] , &s[len]);

  // 
  list<char>::iterator
    where = find(list1.begin(),list1.end(), 'e');
  assert(*where == 'e' && *(++where)=='t');
  cout << *(--where) << *(++where) << endl;
}

