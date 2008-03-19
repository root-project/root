/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>

using namespace std;

int main()
{
  cout << "Demonstrating generic find algorithm with "
    << "a vector." << endl;
  char * s = "C++ is a better C";
  int len = strlen(s);

  cout << "instantiate vector" << endl;

  // vector1 initialization
#if !(G__GNUC>=3)
  vector<char> vector1(&s[0] , &s[len]);
#else
  vector<char> vector1;
  for(int i=0;i<len;i++) vector1.push_back(s[i]);
  vector1.push_back((char)0);
#endif

  // 
  vector<char>::iterator
    where = find(vector1.begin(),vector1.end(), 'e');
  assert(*where == 'e' && *(where+1)=='t');
  cout << *where << *(where+1) << endl;
}

