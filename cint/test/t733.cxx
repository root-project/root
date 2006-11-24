/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "t733.h"
#include <ertti.h>
#include <iostream>
using namespace std;

int main() {
  G__ClassInfo c;
  G__DataMemberInfo m;
  while(c.Next() && strcmp(c.Name(),"bool")!=0 && 
	strcmp(c.Name(),"type_info")!=0) {
    cout << c.Name() << endl;
    m.Init(c);
    while(m.Next()) {
      cout << "    " << m.Type()->Name() << " " << m.Name() << endl;
    }
  }
  return(0);
}
