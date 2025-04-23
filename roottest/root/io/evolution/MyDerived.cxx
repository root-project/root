#include "MyDerived.h"

ClassImp(MyDerived)

// Constructor
MyDerived::MyDerived() : TObject(), fX(0) {}

// Copy constructor
MyDerived::MyDerived(const MyDerived& obj): TObject(obj) {
  fX = obj.fX;
}

// Assignment operator
MyDerived & MyDerived::operator=(const MyDerived& obj) {
  if(this == &obj)return *this;
  
  TObject::operator=(obj);
  fX=obj.fX;
  return *this;
}

// Destructor
MyDerived::~MyDerived() {}

Int_t MyDerived::GetX() const {return fX;}
void MyDerived::SetX(Int_t value) {fX = value;}

