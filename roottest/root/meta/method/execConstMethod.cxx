#include <stdio.h>

class Holder {
   int fValue;
   static const int fConfig;
public:
   int data() const { return fValue; }
   int& data() { return fValue; }
   int const_only() const { return fValue; }
   int& value() { return fValue; }

   static const int& config() { return fConfig; }
};

const int Holder::fConfig = 3;

#include "TClass.h"
#include "TMethod.h"
#include "TList.h"
#include "TInterpreter.h"

int execConstMethod()
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
   if (strcmp(m->GetReturnTypeName(),"int&")!=0) {
      fprintf(stderr,"Did not find the non-const version of data. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }

   m = cl->GetMethod("data","",true);
   if (!m) {
      fprintf(stderr,"Could not find the const version of the data method\n");
      return 1;
   }
   if (strcmp(m->GetReturnTypeName(),"int")!=0) {
      fprintf(stderr,"Did not find the const version of data. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }

   m = cl->GetMethod("const_only","",false);
   if (!m) {
      fprintf(stderr,"Could not find the const_only method given a non-const object.\n");
      return 1;
   }
   if (strcmp(m->GetReturnTypeName(),"int")!=0) {
      fprintf(stderr,"Found a weird version of const_only. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }

   m = cl->GetMethod("const_only","",true);
   if (!m) {
      fprintf(stderr,"Could not find the const version of the const_only method\n");
      return 1;
   }
   if (strcmp(m->GetReturnTypeName(),"int")!=0) {
      fprintf(stderr,"Did not find the right version of const_only. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }

   m = cl->GetMethod("value","",false);
   if (!m) {
      fprintf(stderr,"Could not find the non-const version value method\n");
      return 1;
   }
   if (strcmp(m->GetReturnTypeName(),"int&")!=0) {
      fprintf(stderr,"Did not find the non-const version of value. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }

   m = cl->GetMethod("value","",true);
   if (m) {
      fprintf(stderr,"Found a const version of the value method!. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }

   m = cl->GetMethod("config","");
   if (!m) {
      fprintf(stderr,"Could not find the static method config with a non const search\n");
      return 1;
   }
   if (strcmp(m->GetReturnTypeName(),"const int&")!=0) {
      fprintf(stderr,"Did not find the non-const version of data. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }
   
   m = cl->GetMethod("config","",true);
   if (!m) {
      fprintf(stderr,"Could not find the static method config with a const search\n");
      return 1;
   }
   if (strcmp(m->GetReturnTypeName(),"const int&")!=0) {
      fprintf(stderr,"Did not find the const version of data. Got return type as: %s\n",m->GetReturnTypeName());
      return 1;
   }

   // Trying to execute
   m = TObject::Class()->GetMethod("Print","char*",true);
   if (m) {
      TObject o;
      gInterpreter->Execute(&o,TObject::Class(),m,0);
   } else {
      fprintf(stderr,"Could not find the Print() const method\n");
      return 1;
   }
   return 0;
}
