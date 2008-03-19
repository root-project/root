/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// --------- test.H
class A { 
 public:
  int a;
};
 
enum Op{
  OP_Neg,
  OP_Not,
  OP_Sub};

//A fxx(Op,A,A) ;
A fxx(Op a,A b,A c) { 
  A d;
  switch(a) {
  case OP_Sub:
    d.a = b.a - c.a;
    break;
  }
  return d;
}

inline A operator-(A x,A y){return fxx(OP_Sub,x,y);}

//A fx(Op,A);
A fx(Op a,A b) {
  A d;
  switch(a) {
  case OP_Not:
    d.a = !b.a;
    break;
  case OP_Neg:
    d.a = -b.a;
    break;
  }
  return d;
}
inline A operator-(A x){return fx(OP_Neg,x);}
inline A operator!(A x){return fx(OP_Not,x);}

// ---------- testLinkDef.h
#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class A;
#pragma link C++ function fxx;
#pragma link C++ function operator-(A,A);
#pragma link C++ function fx;
#pragma link C++ function operator-(A);
#pragma link C++ function operator!(A);

#endif


