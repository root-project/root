/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <iostream.h>
#include <deque.h>
#include <algo.h>
#include <assert.h>

main()
{
  deque<int> test;
  //cout << "begin=" << test.begin() << " end=" << test.end() << endl;
  printf("begin=%x end=%x\n",test.begin(),test.end());
}
