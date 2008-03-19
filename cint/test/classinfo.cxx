// classinfo.cxx

//  
//  Check that G__ClassInfo::GetMethod() with a textual
//  argument type list handles an enum type correctly
//  in exact match mode.
//

#include "classinfo.h"
#include <ertti.h>

using namespace std;
using namespace Cint;

void test1()
{
   G__ClassInfo info("A");
   if (info.IsValid()) {
      printf("Got class info for class A.\n");
   }
   else {
      printf("Did not get class info for class A.\n");
   }
   long offset = 0L;
   {
      G__MethodInfo method = info.GetMethod("with_enum", "A::ETags", &offset, G__ClassInfo::ExactMatch);
      if (method.IsValid()) {
         printf("Got method info for A::with_enum(A::ETags).\n");
      }
      else {
         printf("Did not get method info for A::with_enum(A::ETags).\n");
      }
   }
   {
      G__MethodInfo method = info.GetMethod("with_enum", "ETags", &offset, G__ClassInfo::ExactMatch);
      if (method.IsValid()) {
         printf("Got method info for A::with_enum(ETags).\n");
      }
      else {
         printf("Did not get method info for A::with_enum(ETags).\n");
      }
   }
   {
      G__MethodInfo method = info.GetMethod("with_enum", "int", &offset, G__ClassInfo::ExactMatch);
      if (method.IsValid()) {
         printf("Got method info for A::with_enum(int).\n");
      }
      else {
         printf("Did not get method info for A::with_enum(int).\n");
      }
   }
   {
      G__MethodInfo method = info.GetMethod("with_int", "A::ETags", &offset, G__ClassInfo::ExactMatch);
      if (method.IsValid()) {
         printf("Got method info for A::with_int(A::ETags).\n");
      }
      else {
         printf("Did not get method info for A::with_int(A::ETags).\n");
      }
   }
   {
      G__MethodInfo method = info.GetMethod("with_int", "ETags", &offset, G__ClassInfo::ExactMatch);
      if (method.IsValid()) {
         printf("Got method info for A::with_int(ETags).\n");
      }
      else {
         printf("Did not get method info for A::with_int(ETags).\n");
      }
   }
   {
      G__MethodInfo method = info.GetMethod("with_int", "int", &offset, G__ClassInfo::ExactMatch);
      if (method.IsValid()) {
         printf("Got method info for A::with_int(int).\n");
      }
      else {
         printf("Did not get method info for A::with_int(int).\n");
      }
   }
}

int main()
{
   test1();
}

