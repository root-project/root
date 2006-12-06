/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
#include <ertti.h>


class B {
  int sz;
  double *d;
 public:
  //B(int s) : sz(s+1) { printf("B(%d)\n",s); d = new double[sz]; }
  B(int s) { sz=s; printf("B(%d)\n",s); d = new double[sz]; }
  ~B() { printf("~B()\n"); delete[] d; }
  void Set(double offset, double step) {
    printf("Set(%g,%g)\n",offset,step);
    for(int i=0;i<sz;i++) d[i] = offset + step*i;
  }
  void disp() const {
    //void disp() {
    printf("disp() ");
    for(int i=0;i<sz;i++) printf("%g ",d[i]);
    printf("\n");
  }
  double Get(int s) const { return(d[s]); }
  void Set(int s,double v) { d[s]=v; }
};

void test3(const char* name) {
  printf("test3(%s)\n",name);
  long offset;
  G__ClassInfo cl(name+1);
  G__CallFunc ctor;
  G__CallFunc set;
  G__CallFunc disp;
  G__CallFunc dtor;
  ctor.SetFuncProto(&cl,name+1,"int",&offset);
  set.SetFuncProto(&cl,"Set","double,double",&offset);
  disp.SetFuncProto(&cl,"disp","",&offset);
  dtor.SetFuncProto(&cl,name,"",&offset);

  printf("%s %d\n",cl.Name(),cl.IsValid());
  printf("ctor %d\n",ctor.IsValid());
  printf("set %d\n",set.IsValid());
  printf("disp %d\n",disp.IsValid());
  printf("dtor %d\n",dtor.IsValid());

  for(int i=0;i<3;i++) {
    ctor.SetArg((long)(i+5));
    set.SetArg((double)i);
    set.SetArg((double)i/10);
    offset = ctor.ExecInt((void*)0);
    set.Exec((void*)offset);
    disp.Exec((void*)offset);
    dtor.Exec((void*)offset);
    ctor.ResetArg();
    set.ResetArg();
  }
}


int main() {
  //test1("~A");
  //test1("~B");
  //test2("~A");
  //test2("~B");
  //test3("~A");
  test3("~B");
  return 0;
}
