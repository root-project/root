#include "TObject.h"

class Long64_t_Container {
 public:
   /*
   long long i;
   unsigned long long ui;
   long double ld;
   */
   Long64_t i;
   ULong64_t ui;
   long double ld;

   Long64_t geti() { return i; }
   Long64_t_Container() {}
   virtual ~Long64_t_Container() {}
};

