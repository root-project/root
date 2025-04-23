#include "TObject.h"
#include <iostream>

class MyClass : public TObject {
public:
   MyClass() : TObject() {
      ver = 1;
      fArray.emplace_back(1);
      fArray.emplace_back(2);
   }

   void addSomeData() {
      // fArray.push_back(123);
      // fArray.push_back(456);
   }
   void Print(Option_t * /*option*/ ="") const override {
      std::cout << "MyClass::Print ver: " << ver << "\n";
   }


private:
   // rule is only applied if fArray is first!
   int ver;
   std::vector<int> fArray;

   ClassDefOverride(MyClass, 1)
};
