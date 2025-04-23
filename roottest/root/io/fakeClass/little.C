#include "TObject.h"

class little {
   int   var1;
   float var2;
public:
   virtual ~little() {}; // force a virtual table
   float get() { if (var1) return var2; else return 0; }
};

class wrapper : public TObject {
   little e;
   ClassDef(wrapper,1);
};
