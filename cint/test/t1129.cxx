/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <string>
#include <stdio.h>
 
using namespace std;
 
template <typename T> T Foo();
 
template <>
std::string Foo<std::string>()
{
  printf("Foo<std::string>\n");
  return "";
}
 
int main()
{
  // This works:
  Foo<std::string>();
 
  // But this fails:
  Foo<string>();
 
  // cint.exe - application error
  // The exception unknown software exception (0xc00000fd) occurred 
  // in the application at location 0x100b5087.
  return 0;
}

