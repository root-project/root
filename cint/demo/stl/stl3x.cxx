/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream.h>
#include <vector.h>
#include <algo.h>
#include <assert.h>
#include <string.h>

vector<char> vec(char* s)
{
  vector<char> x;
  while(*s != '\0') {
    x.push_back(*s++);
  }
  return x;
}

ostream& operator<<(ostream& os,vector<char>& x) {
  vector<char>::iterator p=x.begin();
  do {
    os << *p++ ;
    //os << ((int)(*p++)) << ' ';
  } while(p!=x.end());
  return(os);
}

int main() 
{
  //test() ; exit();
  cout << "Using reverse algorithm with a vector" << endl;
  vector<char> vector1 = vec("mark twain");
  cout <<  vector1 << endl;
  reverse(vector1.begin(),vector1.end());
  //reverse(&vector1[0],&vector1[vector1.size()]);
  cout <<  vector1 << endl;
  assert(vector1 == vec("niawt kram"));
  if(vector1 == vec("niawt kram")) cout << "true" << endl;
  else                             cout << "false" << endl;
}

test() {
  vector<char> vector1 = vec("mark twain");
  vector<char> vector2 = vec("mark twain");
  cout << vector1 << endl;
  cout << vector2 << endl;
}
