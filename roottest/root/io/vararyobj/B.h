// B.h

#ifndef B_HDR
#define B_HDR

#include "TNamed.h"

class B : public TNamed {
private:
  int           fX;
  double        fY;
public:
  B();
  B(char const *name, char const *title, int x, double y);
  B(const B&);
  B& operator=(const B&);
  ~B() override;
public:
  int           GetX() const     { return fX; }
  double        GetY() const     { return fY; }
  void          SetX(int val)    { fX = val; }
  void          SetY(double val) { fY = val; }
public:
  void          repr() const;
public:
  ClassDefOverride(B,1);
};

#endif // B_HDR
