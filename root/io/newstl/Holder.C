#include "Holder.h"
#include <vector>
void f(int entry = 2) {

   vectorHolder h(2); // Holder< std::vector > h (2);
   h.Verify(entry);
   
}
