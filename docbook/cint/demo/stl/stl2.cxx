/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream.h>
#include <algo.h>
#include <assert.h>
#include <string.h>

int main() 
{
  cout << "Using reverse algorithm with an array" << endl;
  char string1[] = "mark twain";
  cout << string1 << endl;
  int N1 = strlen(string1);
  reverse(&string1[0], &string1[N1]);
  assert(strcmp(string1,"niawt kram")==0);
  cout << string1 << endl;
   
}

