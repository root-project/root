#include "TObject.h"

class little {
   int   var1;
   float var2;
public:
   virtual ~little() {}; // force a virtual table
};

class wrapper : public TObject {
   little e;
   ClassDef(wrapper,1);
};
