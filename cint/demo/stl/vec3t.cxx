/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
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
  vector<U*> vector1;
  const int N=10;
  U* xx;
  for(int k=0;k!=N;++k) {
    //xx = new U(k);
    //vector1.push_back(xx);
    vector1.push_back(new U(k));
  }

  vector<U*>::iterator u=vector1.begin();
  while(u!=vector1.end()) {
    cout << "vector1 " << (*u)->id << endl;
    u++;
  }

  cout << "Using generic reverse algorithm with a vector "
       << "of user-defined objects" << endl;

  reverse(vector1.begin(),vector1.end());
  u=vector1.begin();
  while(u!=vector1.end()) {
    cout << "vector1 " << (*u)->id << endl;
    u++;
  }

}

