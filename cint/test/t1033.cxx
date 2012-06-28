/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <ertti.h>

class A {
public:
  long long a;
  unsigned long long b;
  long double c;
};

int main() {
  G__TypeInfo ll("long long");
  G__TypeInfo ull("unsigned long long");
  G__TypeInfo ld("long double");
  printf("long long  %x\n",ll.Property());
  printf("unsigned long long  %x\n",ull.Property());
  printf("long double  %x\n",ld.Property());
  G__ClassInfo a("A");
  G__DataMemberInfo d(a);
  while(d.Next()) {
    printf("%s  %d\n",d.Name(),d.Property());
  }
  return 0;
}
