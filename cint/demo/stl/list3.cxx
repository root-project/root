/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// still have problem running this program
#include <stdio.h>
#include <iostream.h>
#include <_list.h> // problem in precompiled list class and cint
#include <algo.h>
#include <assert.h>

class U {
public:
 unsigned long id; 
 U() : id(0) { printf("U(%d)\n",id); }
 U(unsigned long x) : id(x) { printf("U(%d)\n",id);  }
 //U(U& obj) : id(obj.id) { }
};


bool operator==(const U& x, const U& y)
{
  return x.id==y.id;
}

int main()
{
  list<U> list1,list2;
  const int N=10;
  for(int k=0;k!=N;++k) list1.push_back(U(k));
  for(int k=0;k!=N;++k) list2.push_back(U(N-1-k));

  cout << "Using generic reverse algorithm with a list "
       << "of user-defined objects" << endl;

  reverse(list1.begin(),list1.end());

  if(list1==list2) cout << "true" << endl;
  else               cout << "false" << endl;
  assert(list1==list2);
}

