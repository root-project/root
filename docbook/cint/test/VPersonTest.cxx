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
#else // INTERP
#include "VPerson.dll"
#endif // INTERP

#define NUM 5

void test1()
{
   VArray a;
   VPerson* p = 0;
   VCompany* p1 = 0;
   for (int i = 0; i < NUM; ++i) {
      if (i % 2) {
         p = new VPerson("name", i);
         a.Add(p, -1);
      }
      else {
         p1 = new VCompany("company", i);
         a.Add(p1, -1);
      }
   }
   for (int i = 0; i < NUM; ++i) {
      a[i].disp();
   }
   for (int i = 0; i < NUM; ++i) {
      delete a.Delete(-1, 0);
   }
}

void test2()
{
   VArray a;
   for (int i = 0; i < NUM; ++i) {
      if (i % 2) {
         a.Add(new VPerson("name", i), -1);
      }
      else {
         a.Add(new VCompany("company", i), -1);
      }
   }
   for (int i = 0; i < NUM; ++i) {
      a[i].disp();
   }
   for (int i = 0; i < NUM; ++i) {
      a.Delete(-1, 1);
   }
}

int main()
{
   test1();
   test2();
   return 0;
}

