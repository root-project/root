/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <cstdio>

using namespace std;

void fff(const char* foobar)
{
   printf("fff(%s)\n", foobar);
}

void aaa(const char* foobar)
{
   printf("aaa(%s)\n", foobar);
}

void ccc(const char* foobar)
{
   printf("ccc(%s)\n", foobar);
}

void ddd(const char* foobar)
{
   printf("ddd(%s)\n", foobar);
}

void foo(const char* foobar)
{
   printf("foo(%s)\n", foobar);
}

int main()
{
   foo("abc");
   aaa("abc");
   ccc("abc");
   ddd("abc");
   fff("abc");
   return 0;
}

