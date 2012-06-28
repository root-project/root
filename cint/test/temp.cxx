/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#define INTERP
#ifdef INTERP
//#include "VObject.cxx"
//#include "VPerson.cxx"
//#include "VCompany.cxx"
//#include "VArray.cxx"
//#include "VString.cxx"
#else
#include "VPerson.dll"
#endif

#define NUM 0

void test0() {
  VArray a;
  VPerson* p;
  VCompany* p1;
  Int_t i;

#if 0
  p=new VPerson("name0",i);
  a.Add(p,-1);
  p1=new VCompany("company1",i);
  a.Add(p1,-1);

  for(i=0;i<NUM;i++) {
    a[i].disp();
  }  

  for(i=0;i<NUM;i++) {
    delete a.Delete(-1,0);
  }  
#endif
}

int main() {
  //test0();
  return 0;
}


