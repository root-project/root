/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#ifdef __CINT__
#include <ertti.h>
#else
#include <Api.h>
#endif

template<class T,class B ,class C = char > class A {
  T* t;
  B* b;
  C* c;
};

// A<double> a1;
A<double,short> a2;
A<double,short,float> a3;

template<class T,template<class U> class S> class X {
  S<T,unsigned char> s1;
};

X<long,A> a4;


main() {
  G__ClassInfo c1("A<double,short>"); disp(c1);
  G__ClassInfo c2("A<double,short,float>"); disp(c2);
  //G__ClassInfo c3old("X<long,A>::A<long,unsigned char>"); disp(c3old);
  G__ClassInfo c3old("A<long,unsigned char>"); disp(c3old);
  G__ClassInfo c3new("A<long,unsigned char>"); disp(c3new);
  G__ClassInfo c4("X<long,A>"); disp(c4);
}

void disp(G__ClassInfo& c)
{
  G__DataMemberInfo d(c);
  while(d.Next()) {
    printf("%s %s\n",d.Type()->Name(),d.Name());
  }
}
