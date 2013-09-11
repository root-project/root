// NO #include "Template.h"
class TagClassWithoutDefinition;
typedef TagClassWithoutDefinition Tag_t;
void runtemplateAutoload() {
   Template<float,0,Tag_t> inTmpltFloat_C;
   Template<int,0,Tag_t> inTmpltInt0_C;
   Template<int,1,Tag_t> inTmpltInt1_C;

   ANamespace::Template<int,0,Tag_t> inANSTmpltInt0_C;
   ANamespace::Template<int,1,Tag_t> inANSTmpltInt0_C_too;
}
