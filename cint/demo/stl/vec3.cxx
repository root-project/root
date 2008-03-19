/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <iostream.h>
#include <vector.h>
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
  vector<U> vector1,vector2;
  const int N=10;
  for(int k=0;k!=N;++k) vector1.push_back(U(k));
  for(int k=0;k!=N;++k) vector2.push_back(U(N-1-k));

  cout << "Using generic reverse algorithm with a vector "
       << "of user-defined objects" << endl;

  reverse(vector1.begin(),vector1.end());

  if(vector1==vector2) cout << "true" << endl;
  else                 cout << "false" << endl;
  assert(vector1==vector2);
}

