#ifndef MYDERIVED_H
#define MYDERIVED_H

#include <TObject.h>

class MyDerived: public TObject {

public:

  // Constructors, assignment operator, destructor
  MyDerived();
  MyDerived(const MyDerived& obj);
  MyDerived & operator=(const MyDerived& obj);
  ~MyDerived() override;

  // Access methods
  Int_t GetX() const;
  void SetX(Int_t value);

private:

  Int_t fX; //

  ClassDefOverride(MyDerived,2)

};

#endif
