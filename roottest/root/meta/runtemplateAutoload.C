// NO #include "Template.h"
class TagClassWithoutDefinition;
typedef TagClassWithoutDefinition Tag_t;
void runtemplateAutoload() {
   Template<float,0,Tag_t,float,0> inTmpltFloat_C;
   Template<int,0,Tag_t,int,0> inTmpltInt0_C;
   Template<int,1,Tag_t,int,1> inTmpltInt1_C;
   //Template<short, 0, Tag_t, int, 1> inTmpltNoSpec_C;
   TClass* cl = TClass::GetClass("Template<short, 0, Tag_t, int, 1>", kTRUE /*load*/);
   if (!cl) printf("No TClass for Template<short, 0, Tag_t, int, 1>!\n");
   else printf("The TClass for Template<short, 0, Tag_t, int, 1> is%s loaded\n",
               (cl->IsLoaded() ? "": " not"));

   ANamespace::Template<int,0,Tag_t> inANSTmpltInt0_C;
   ANamespace::Template<int,1,Tag_t> inANSTmpltInt0_C_too;
}
