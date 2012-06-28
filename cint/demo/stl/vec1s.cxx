/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>

int main()
{
  std::cout << "Demonstrating generic find algorithm with "
    << "a vector." << std::endl;
  char * s = "C++ is a better C";
  int len = std::strlen(s);

  std::cout << "instantiate vector" << std::endl;

  // vector1 initialization
#if !(G__GNUC>=3)
  std::vector<char> vector1(&s[0] , &s[len]);
#else
  vector<char> vector1;
  for(int i=0;i<len;i++) vector1.push_back(s[i]);
  vector1.push_back((char)0);
#endif

  // 
  std::vector<char>::iterator
    where = std::find(vector1.begin(),vector1.end(), 'e');
  std::assert(*where == 'e' && *(where+1)=='t');
  std::cout << *where << *(where+1) << std::endl;
}

