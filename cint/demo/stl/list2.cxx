/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <iostream.h>
#include <_list.h> // problem in precompiled list class and cint
#include <algo.h>
#include <assert.h>


list<char> vec(char* s)
{
  list<char> x;
  while(*s != '\0') {
    x.push_back(*s++);
  }
  return x;
}

ostream& operator<<(ostream& os,list<char>& x) {
  list<char>::iterator p=x.begin();
  do {
    os << *p++ ;
    //os << ((int)(*p++)) << ' ';
  } while(p!=x.end());
  return(os);
}

int main() 
{
  //test() ; exit();
  cout << "Using reverse algorithm with a list" << endl;
  list<char> list1 = vec("mark twain");
  cout <<  list1 << endl;
  reverse(list1.begin(),list1.end());
  cout <<  list1 << endl;
  assert(list1 == vec("niawt kram"));
  if(list1 == vec("niawt kram")) cout << "true" << endl;
  else                             cout << "false" << endl;
}

test() {
  list<char> list1 = vec("mark twain");
  list<char> list2 = vec("mark twain");
  cout << list1 << endl;
  cout << list2 << endl;
}
