/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <iostream.h>
#include <list.h> // problem in precompiled list class and cint
#include <algo.h>
#include <assert.h>

main()
{
  list<int> test;
  //cout << "begin=" << test.begin() << " end=" << test.end() << endl;
  printf("begin=%x end=%x\n",test.begin(),test.end());
}
