/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 021220regexp.txt, Philippe

#include <list>
using namespace std;
class BaseStar {
//...
};

template<class Star> class StarList : public list <Star*> {
//...
};

typedef list<BaseStar*> blist;

#ifdef __CINT__
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;
#pragma link C++ class BaseStar;
#pragma link C++ class StarList<BaseStar>;
#pragma link C++ class list<BaseStar*>;
//typedef list<BaseStar*> blist;
#pragma link C++ class blist::iterator;
#pragma link C++ class list<BaseStar*>::iterator;
#endif

#include <stdio.h>
void test() {
  printf("success\n");
}

