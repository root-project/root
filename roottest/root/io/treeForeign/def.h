#include "TClonesArray.h"

class MyClass {
 public:
   int myvar;
   MyClass() : myvar(-1) {};
   MyClass(int v) : myvar(v) {};
};

class Wrapper : public TObject {
 public:
   MyClass *chunk;
   Wrapper() : chunk(new MyClass) {};
   Wrapper(int v) : chunk(new MyClass(v)) {};
   ~Wrapper() { delete chunk; }

   ClassDef(Wrapper,1);
};

