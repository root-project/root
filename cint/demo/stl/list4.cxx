/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// still have problem running this program
#include <iostream.h>
#include <_list.h> // problem in precompiled list class and cint
#include <algo.h>

int main()
{
  cout << "This program still have problem running on cint" << endl;
  list<int> list1,list2;
  const int N=10;
  for(int k=0;k!=N;++k) list1.push_back(rand()%10);
  list<int>::iterator it = list1.begin();
  for(int k=0;k!=N;++k) cout << *it++ << " ";
  cout << endl;

#if 0
  list1.sort(); // PROBLEM HAPPENS HERE
#else
  reverse(list1.begin(),list1.end());
#endif
  
  list<int>::iterator it2 = list1.begin();
  for(int k=0;k!=N;++k) cout << *it2++ << " ";
  cout << endl;
}

