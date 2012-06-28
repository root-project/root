/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "t733.h"
#include <ertti.h>
#include <iostream>
using namespace std;

int main() 
{
  cout << endl;
  G__ClassInfo c;
  G__DataMemberInfo m;
#if (G__CINTVERSION > 70000000))
  // Skip global namespace
  c.Next();
  // Skip bytecode arena
  c.Next();
#endif
#ifdef __APPLE__
  // Skip va_list on macos
  c.Next();
#endif
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
