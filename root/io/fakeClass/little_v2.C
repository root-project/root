#include "TObject.h"

class little {
   int   var1;
   float var2;
   int var3;
public:
   virtual ~little() {}; // force a virtual table
};

class wrapper : public TObject {
   little e;
   int var4;
   ClassDef(wrapper,1); // Intentionally forget to change version number
};
