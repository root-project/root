/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1030.h"
#endif

using namespace Experiment1;

int main() {
  Experiment1::TExperimentalDataNode a;
  Experiment1::TExperimentalData b1;
  Experiment1::TExperimentalData b2;

  std::vector<Experiment1::TExperimentalData> c;
  std::vector<std::vector<Experiment1::TExperimentalData> > d;
  std::vector<std::vector<std::vector<Experiment1::TExperimentalData> > > e;

  std::map<std::string,Experiment1::TExperimentalData> f;
  std::map<std::string,std::vector<Experiment1::TExperimentalData> > g;
  std::map<std::string,std::vector<std::vector<Experiment1::TExperimentalData> > > h;
#if 0
  std::map<std::string,std::vector<std::vector<std::vector<Experiment1::TExperimentalData> > > > i;
#endif

  c.push_back(b1);
  c.push_back(b2);
  d.push_back(c);
  e.push_back(d);

  f["b1"]=b1;
  f["b2"]=b2;
  g["c"]=c;
  h["d"]=d;
#if 0
  i["e"]=e;
#endif

  f["b1"];
  f["b2"];
  g["c"];
  h["d"];
#if 0
  i["e"];
#endif

  printf("Success\n");
  return 0;
}
