////////////////////////////////////////////////////////////////////////////////
// This test case assures that TCallFunc works used through TInterpreter's    //
// and TMethodCall's interfaces.                                              //
//                                                                            //
// It tests variety of scenarios:                                             //
//    - inlined                                                               //
//    - outlined                                                              //
//    - static                                                                //
////////////////////////////////////////////////////////////////////////////////
// Author: Vassil Vassilev

extern "C" int printf(const char* fmt, ...);

class A {
private:
   float a;
public:
   A() : a(4.f){}
   void inlineNoArgsNoReturn() { printf("A::inlineNoArgsNoReturn\n");}
   void outlinedNoArgsNoReturn();
};
void A::outlinedNoArgsNoReturn() { printf("A::outlinedNoArgsNoReturn\n");}



#include "TClass.h"
#include "TInterpreter.h"

//TODO: Extend

void runAllThroughTInterpreterInterfaces() {
   printf("Running through TInterpreter public interfaces...\n");

   ClassInfo_t* tagX = 0;
   void* tagXAddr = 0;
   CallFunc_t* tagXCtor = gInterpreter->CallFunc_Factory();
   CallFunc_t* tagXDtor = gInterpreter->CallFunc_Factory();
   CallFunc_t* mc = gInterpreter->CallFunc_Factory();
   Longptr_t offset = 0;

   // Construct TClass for A
   tagX = TClass::GetClass("A")->GetClassInfo();
   // Create an instance of A in memory
   gInterpreter->CallFunc_SetFuncProto(tagXCtor, tagX, "A", "", &offset);
   // The interpreter owns temporaries, don't delete.
   tagXAddr = (void*)gInterpreter->CallFunc_ExecInt(tagXCtor, /*address*/(void*)0);

   // Run A::inlineNoArgsNoReturn()
   gInterpreter->CallFunc_SetFuncProto(mc, tagX, "inlineNoArgsNoReturn", "", &offset);
   gInterpreter->CallFunc_Exec(mc, tagXAddr);

   // Run A::outlinedNoArgsNoReturn()
   gInterpreter->CallFunc_SetFuncProto(mc, tagX, "outlinedNoArgsNoReturn", "", &offset);
   gInterpreter->CallFunc_Exec(mc, tagXAddr);


   // Cleanup
   gInterpreter->CallFunc_Delete(mc);
   gInterpreter->CallFunc_Delete(tagXCtor);
}

#include "TMethodCall.h"

void runAllThroughTMethodCall() {
   printf("Running through TMethodCall...\n");
   TMethodCall method;
   Long_t result_long = 0;
   Double_t result_double;

}

void runmemberFunc() {
   runAllThroughTInterpreterInterfaces();
   printf("======================================================\n");
   runAllThroughTMethodCall();
}
