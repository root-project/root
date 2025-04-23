#ifndef __CINT__
#include "Params.h"
#endif

void testDefaultParams(const char* mode) {
   // One object of type Base and one of type derived
   Base    *base    = new Base(mode);
   Derived *derived = new Derived(mode);
   
   // Another of type Base pointing to a Derived 
   Base    *base_der = derived;

   // Default parameter of the base funtion
   base->FunctionX();

   // Default parameter of the derived funtion
   derived->FunctionX();

   // Default parameter of the base funtion again!!!
   // Note: they are evaluated according to their static
   // type since in this case it's "Base *base_der"
   base_der->FunctionX();


   base->FunctionY(); // counter: 1
   derived->FunctionY(); // counter: 2
   base_der->FunctionY(); // counter: 3

   base_der->FunctionY(42, 42); // counter: unchanged!

   base->FunctionY(); // counter: 4
   derived->FunctionY(); // counter: 5
   base_der->FunctionY(); // counter: 6

   delete base;
   delete derived;
}
