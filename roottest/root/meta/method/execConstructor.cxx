#include <stdio.h>

class Holder {
public:
   Holder(const char*) {}
   Holder() {}
   Holder(int &) {}
   ~Holder() {}
   
   int data() { return 0; }
   operator int() const { return 0; }
};

#include "TClass.h"
#include "TMethod.h"
#include "TList.h"
#include "TInterpreter.h"

int execConstructor()
{
   TClass *cl = TClass::GetClass("Holder");
   if (!cl) {
      fprintf(stderr,"Could not find the TClass for Holder\n");
      return 1;
   }
   // cl->GetListOfMethods()->ls();

   TMethod * m = cl->GetMethod("data","");
   if (!m) {
      fprintf(stderr,"Could not find the non-const version data method\n");
      return 1;
   }
   if (strcmp(m->GetReturnTypeName(),"int")!=0) {
      fprintf(stderr,"Did not find the non-const version of data. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }
   if ( m->ExtraProperty() & kIsConstructor) {
      fprintf(stderr,"mistake the data member function for a constructor\n");
      return 1;
   }

   m = cl->GetMethod("Holder","",true);
   if (!m) {
      fprintf(stderr,"Could not find Holder() \n");
      return 1;
   }
   if ( !(m->ExtraProperty() & kIsConstructor)) {
      fprintf(stderr,"mistake Holder() for a non-constructor\n");
      return 1;
   }

   m = cl->GetMethodWithPrototype("Holder","const char*",false);
   if (!m) {
      fprintf(stderr,"Could not find Holder(const char*) \n");
      return 1;
   }
   if ( !(m->ExtraProperty() & kIsConstructor)) {
      fprintf(stderr,"mistake Holder(const char*) for a non-constructor\n");
      return 1;
   }

   m = cl->GetMethodWithPrototype("Holder","int&",true);
   if (!m) {
      fprintf(stderr,"Could not find Holder() \n");
      return 1;
   }
   if ( !(m->ExtraProperty() & kIsConstructor)) {
      fprintf(stderr,"mistake Holder(int&) for a non-constructor\n");
      return 1;
   }

   m = cl->GetMethodWithPrototype("~Holder","",true);
   if (!m) {
      fprintf(stderr,"Could not find ~Holder() \n");
      return 1;
   }
   if ( !(m->ExtraProperty() & kIsDestructor)) {
      fprintf(stderr,"mistake ~Holder() for a non-destructor\n");
      return 1;
   }
   
   m = cl->GetMethodWithPrototype("operator int","",true);
   if (!m) {
      fprintf(stderr,"Could not find operator int() \n");
      return 1;
   }
   if ( !(m->ExtraProperty() & kIsConversion)) {
      fprintf(stderr,"mistake Holder(int&) for a non-conversion\n");
      return 1;
   }

   // Trying to execute
//   m = TObject::Class()->GetMethod("Print","char*",true);
//   if (m) {
//      TObject o;
//      gInterpreter->Execute(&o,TObject::Class(),m,0);
//   } else {
//      fprintf(stderr,"Could not find the Print() const method\n");
//      return 1;
//   }
   return 0;
}
