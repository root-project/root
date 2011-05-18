/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
class Simple {
 public:
  Simple() { printf("\nSimple.C start\n"); }
  Description() { 
    printf("Cint is interpreting cint itself, "); 
    printf("and the interpreted cint interprets Simple.C.\n"); 
  }
  Loop() {
    printf("Loop test\n");
    for(int i=0;i<5;i++) {
      printf("%d*3.14=%g\n",i,3.14*i);
    }
  }
  ~Simple() { printf("End\n"); }
};


int main() {
  Simple a;
  a.Description();
  a.Loop();
  return 0;
}
