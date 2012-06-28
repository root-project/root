/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <cstdio>
#include <cstring>

using namespace std;

void do_assign(char* a, char** b)
{
   *b = a;
}

int main()
{
   char buf[2048];
   buf[0] = '\0';
   char* a = buf;
   strcpy(a, "test");
   char* b = 0;

   do_assign(a, &b);
   printf("%s\n", a);
   printf("%s\n", b);

   return 0;
}

