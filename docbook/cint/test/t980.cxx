/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t980.h"
#endif

#include <cstdio>
#include <cstring>

using namespace std;

int main(int, char**)
{
   A b;
   b = A("A part") + " of a whole";
   A a = A("A part") + " of a whole";
   printf("%s. %s.\n", a.val(), b.val());

   f(a, "A part of a whole");
   f("A part of a whole", a);

   if (!strcmp(a, "A part of a whole")) {
      printf("true\n");
   }
   else {
      printf("false\n");
   }
   if (!strcmp(a, "a part of a whole")) {
      printf("true\n");
   }
   else {
      printf("false\n");
   }

   if (!strcmp(a.val(), "A part of a whole")) {
      printf("true\n");
   }
   else {
      printf("false\n");
   }
   if (!strcmp(a.val(), "a part of a whole")) {
      printf("true\n");
   }
   else {
      printf("false\n");
   }
   return 0;
}

