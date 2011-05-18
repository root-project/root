/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <iostream>
#include "qtcint.dll"

ostream& operator<<(ostream& ost,const QString& x) {
  ost << x.data();
  return(ost);
}

int main() {
  QString a("abc");
  QString b("def");
  QString c("def");
  a += b;
  cout << a.data() << " " << b.data() << endl;
  cout << a << " " << b << endl;
  return 0;
}
