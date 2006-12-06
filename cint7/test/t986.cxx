/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 030115string 030115string2
// Problem with const static 

//#include <string.dll>
#include <string>
#include <iostream>
using namespace std;
#ifndef G__ROOT
//#include "t986.dll"
typedef char Text_t;
#endif

//#define TEST
#ifdef TEST
string fname_long(const Text_t *a){
  string b="bbb";
  b += a;
  return b;
}
#endif

void t986(void){
  int i;
  const int maxsize=10;
  static double x[maxsize];
  static const string name="t986";

  for(i=0;i<maxsize;i++) x[i] = i*1.2;
  const string a(fname_long("eta"));   // fine

  cout << "maxsize=" << maxsize << " : " ;
  for(i=0;i<maxsize;i++) cout << x[i] << " " ;
  cout << endl;
  
  cout << "name=" << name << endl;
  cout << a << endl;
}

#ifndef TEST
string fname_long(const Text_t *a){
  string b="bbb";
  b += a;
  return b;
}
#endif

#ifndef G__ROOT
int main() {
  t986();
  return 0;
}
#endif
