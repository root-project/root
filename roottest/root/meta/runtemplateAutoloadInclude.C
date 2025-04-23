#include "Template.h"
#include "runtemplateAutoload.C"

// Even with the definition of the template the libraries for the
// instances must be autoloaded.

void runtemplateAutoloadInclude() {
   Template<float,1,Tag_t> nowhere_T_F1;

   // Run the same test as runtemplateAutoloadInclude.C:
   runtemplateAutoload();

   Template<double,1,Tag_t> nowhere_T_D1;
   ANamespace::Template<int,3,Tag_t> nowhere_ANST_I3;
   ANamespace::Template<float,3,Tag_t> nowhere_ANST_F3;
}
