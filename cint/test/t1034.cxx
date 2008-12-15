/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// the old reference output comes from Cygwin. mismatch may be observed with
// other environment.

#include <ertti.h>

class A {
public:
  long long a;
  unsigned long long b;
  long double c;
};

int main() {
  for(int i=0;i<2;i++) {
    
    // typeid
    printf("typeid(long long).name()=%s\n",typeid(long long).name());
    printf("typeid(unsigned long long).name()=%s\n",typeid(unsigned long long).name());
    printf("typeid(long double).name()=%s\n",typeid(long double).name());
    
    // G__TypeInfo
    G__TypeInfo ll("long long");
    G__TypeInfo ull("unsigned long long");
    G__TypeInfo ld("long double");
    printf("'%s' '%s' %x\n",ll.Name(),ll.TrueName(),ll.Property());
    printf("'%s' '%s' %x\n",ull.Name(),ull.TrueName(),ull.Property());
    printf("'%s' '%s' %x\n",ld.Name(),ld.TrueName(),ld.Property());
  }

  // G__DataMemberInfo
  G__ClassInfo a("A");
  G__DataMemberInfo d(a);
  while(d.Next()) {
    printf("%s '%s' '%s' %x\n",d.Name(),d.Type()->Name(),d.Type()->TrueName(),d.Property());
  }
  return 0;
}
