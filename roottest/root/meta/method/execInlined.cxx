#include <stdio.h>

class MyClass {
   int fValue;
public:
   int data() { return fValue; }
   inline int dataExplInline() { return fValue; }
   template <typename T> int dataTempl() { return fValue; }
   template <typename T> inline int dataTemplExplInline() { return fValue; }

   int dataNotInline();
};

int MyClass::dataNotInline()
{
   return fValue;
}

#include "TClass.h"
#include "TMethod.h"
#include "TFunctionTemplate.h"

int execInlined()
{
   TClass *cl = TClass::GetClass("MyClass");
   if (!cl) {
      fprintf(stderr, "Could not find the TClass for MyClass\n");
      return 1;
   }

   TMethod *m = cl->GetMethod("data", "");
   if (!m) {
      fprintf(stderr, "Could not find the data method\n");
      return 1;
   }
   if (!(m->ExtraProperty() & kIsInlined)) {
      fprintf(stderr, "The data method should have the inlined property\n");
      return 1;
   }

   m = cl->GetMethod("dataExplInline", "");
   if (!m) {
      fprintf(stderr, "Could not find the dataExplInline method\n");
      return 1;
   }
   if (!(m->ExtraProperty() & kIsInlined)) {
      fprintf(stderr, "The dataExplInline method should have the inlined property\n");
      return 1;
   }

   TFunctionTemplate *ft = cl->GetFunctionTemplate("dataTempl");
   if (!ft) {
      fprintf(stderr, "Could not find the dataTempl method template\n");
      return 1;
   }
   if (!(ft->ExtraProperty() & kIsInlined)) {
      fprintf(stderr, "The dataTempl method template should have the inlined property\n");
      return 1;
   }

   ft = cl->GetFunctionTemplate("dataTemplExplInline");
   if (!ft) {
      fprintf(stderr, "Could not find the dataTemplExplInline method template\n");
      return 1;
   }
   if (!(ft->ExtraProperty() & kIsInlined)) {
      fprintf(stderr, "The dataTemplExplInline method template should have the inlined property\n");
      return 1;
   }

   m = cl->GetMethod("dataNotInline", "");
   if (!m) {
      fprintf(stderr, "Could not find the dataNotInline method\n");
      return 1;
   }
   if (m->ExtraProperty() & kIsInlined) {
      fprintf(stderr, "The dataNotInline method should not have the inlined property\n");
      return 1;
   }

   return 0;
}
