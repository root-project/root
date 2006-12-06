/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <stdlib.h>

void assign(char *a, char **b)
{
  *b = a;
}

int main() {
  char *a = "test";
  char *b = NULL; 
    
  assign(a,&b);
  puts(a);
  puts(b);
  return 0;
}

