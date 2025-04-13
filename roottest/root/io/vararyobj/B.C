// B.C

#include "B.h"
#include "TNamed.h"

#include <iostream>

using namespace std;

ClassImp(B);

B::B() : TNamed(), fX(10), fY(20.0) {}

B::B(char const* name, char const* title, int x, double y) : TNamed(name, title), fX(x), fY(y) {}

B::~B() {}

B::B(const B& b) : TNamed(b), fX(b.fX), fY(b.fY) {}

B& B::operator=(const B& b) {
  if (&b != this) {
    fX = b.fX;
    fY = b.fY;
  }
  return *this;
}

void B::repr() const
{
  cout << GetName() << " " << GetTitle() << " fX: " << fX << " fY: " << fY << endl;
}

