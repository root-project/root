#include "TObject.h"

class little {
   int   var1;
   float var2;
   int var3;
public:
   float get() { if (var1) return var2; else return (float)var3; }
   virtual ~little() {}; // force a virtual table
};

class wrapper : public TObject {
   little e;
   int var4;
   ClassDef(wrapper,1); // Intentionally forget to change version number
public:
   int getVar4() { return var4; }
};
