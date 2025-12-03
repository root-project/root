#ifndef B_H
#define B_H

#include "TObject.h"

class B : public TObject {
public:
  B() : fNum(0) {}
  int  GetNum() const { return fNum; }
  void SetNum(int num) { fNum = num; }
private:
  int fNum;

ClassDef(B,1)

};
#endif  //B_H

#ifndef A_H
#define A_H

#include "TObject.h"
/* #include "b.h" */

class A : public TObject {
public:
  A():fB() {}
  const B& GetB() const { return fB; }
private:
  const B fB;

ClassDef(A,1)

};
#endif  //A_H

// and b.h:-

#ifdef __MAKECINT__
#pragma link C++ class A+;
#pragma link C++ class B+;
#endif

