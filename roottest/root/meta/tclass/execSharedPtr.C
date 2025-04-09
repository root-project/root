#include <memory>
#include <string>
#include "TError.h"
#include "TMethod.h"
#include "TMethodCall.h"

int execSharedPtr()
{
#if __cplusplus >= 201103L
   string output;
   gInterpreter->GetInterpreterTypeName("std::__shared_ptr<int>",output,kTRUE);
   if ( output.length() && output != "__shared_ptr<int>" ) {
      Error("GetInterpreterTypeName","Instead of __shared_ptr<int>, we got %s.",output.c_str());
      //return 1;
   }
   TClass *c = TClass::GetClass("std::shared_ptr<int>");
   if (!c) {
      Error("GetClass","Did not find the TClass for std::shared_ptr<int>.");
      return 2;
   }
   TObject *m = c->GetListOfAllPublicMethods()->FindObject("get");
   if (!m) {
      Error("GetListOfAllPublicMethods","Did not find the get method");
      return 3;
   }
   if (0 != strcmp("get",m->GetName())) {
      Error("TMethod","Instead of get the name of the method is %s",m->GetName());
      return 4;
   }
   std::shared_ptr<int> ptr;
   TMethodCall call((TFunction*)m);
   call.Execute();
   
   c = TClass::GetClass("__shared_ptr<int>");
   if (!c) c = TClass::GetClass("std::shared_ptr<int>");
   if (c) {
      m = c->GetListOfAllPublicMethods()->FindObject("operator=");
      if (!m) {
         Error("GetListOfAllPublicMethods","Did not find the operator= method");
         return 3;
      }
      if (0 != strcmp("operator=",m->GetName())) {
         Error("TMethod","Instead of operator= the name of the method is %s",m->GetName());
         return 4;
      }
      TMethodCall callop((TFunction*)m);
      callop.Execute(&ptr);

      m = c->GetListOfAllPublicMethods()->FindObject("swap");
      if (!m) {
         Error("GetListOfAllPublicMethods","Did not find the swap method");
         return 3;
      }
      if (0 != strcmp("swap",m->GetName())) {
         Error("TMethod","Instead of swap the name of the method is %s",m->GetName());
         return 4;
      }
      TMethodCall callswap((TFunction*)m);
      callswap.Execute(&ptr);
   }
#else
   // Emulated expected errors
  Error("TClingCallFunc::exec","The method get is called without an object.");
  Error("TClingCallFunc::exec","Not enough arguments provided for operator= (0 instead of the minimum 1)");
  Error("TClingCallFunc::exec","Not enough arguments provided for swap (0 instead of the minimum 1)");

#endif
   return 0;
}
