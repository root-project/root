#include "ROOT10426.h"

int execROOT10426() {
   // With modules, the following line triggers massive deserialization
   // and in turn jitting of static initialization. As this happens during
   // the jitting of this function, this needs recursive jitting, memory
   // finalization of the inner jitted memory (but not the outer which
   // still has relocations to be applied!) and then invocation of the
   // jitted code.
   // If the call fails on the first assember instruction, the memory
   // was likely not finalized. If the outer jit fails with not being able
   // to write a relocation then the inner jit has finalized not only its
   // code memory but also the outer jit's code memory.
   if (AutoloadMe{}.foo() != 42)
      return 1;
   return 0;
}
