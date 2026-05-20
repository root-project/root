
#include "TObject.h"
#include <iostream>

class MyClass : public TObject {
public:
   // MyClass() : TObject() {ver = 2; fArray[0] = fArray[1] = -1; }
   MyClass() : TObject() {
      ver = 2;
   }

   void addSomeData() {}
   void Print(Option_t* /*option=""*/) const override {
      std::cout << "MyClass::Print ver: " << ver << "\n";
      //std::cout << "arr[0]: " << fArray[0] << "\n";
      //std::cout << "arr[1]: " << fArray[1] << "\n";
   }


private:
   int ver;
   int fArray[2];

   ClassDefOverride(MyClass, 2)
};
