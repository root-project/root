/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#define INTERP
#ifdef INTERP
#include "VObject.cxx"
#include "VPerson.cxx"
#include "VCompany.cxx"
#include "VArray.cxx"
#include "VString.cxx"
#else
#include "VPerson.dll"
#endif

#define NUM 5

#if 0
void test0() {
  VArray a;
  VPerson* p;
  VCompany* p1;
  Int_t i;
  p=new VPerson("name0",i);
  a.Add(p,-1);
  p1=new VCompany("company1",i);
  a.Add(p1,-1);
  p=new VPerson("name2",i);
  a.Add(p,-1);
  p1=new VCompany("company3",i);
  a.Add(p1,-1);
  p=new VPerson("name4",i);
  a.Add(p,-1);

  for(i=0;i<NUM;i++) {
    a[i].disp();
  }  

  for(i=0;i<NUM;i++) {
    delete a.Delete(-1,0);
  }  
}
#endif

void test1() {
  VArray a;
  VPerson* p;
  VCompany* p1;
  Int_t i;
  for(i=0;i<NUM;i++) {
    if(i%2) {
      p=new VPerson("name",i);
      a.Add(p,-1);
    }
    else {
      p1=new VCompany("company",i);
      a.Add(p1,-1);
    }
  }  

  for(i=0;i<NUM;i++) {
    a[i].disp();
  }  

  for(i=0;i<NUM;i++) {
    delete a.Delete(-1,0);
  }  
}

void test2() {
  VArray a;
  //VPerson* p;
  Int_t i;
  for(i=0;i<NUM;i++) {
    if(i%2) a.Add(new VPerson("name",i),-1);
    else    a.Add(new VCompany("company",i),-1);
  }  

  for(i=0;i<NUM;i++) {
    a[i].disp();
  }  

  for(i=0;i<NUM;i++) {
    a.Delete(-1,1);
  }  
}

int main() {
  //test0();
  test1();
  test2();
  return 0;
}
