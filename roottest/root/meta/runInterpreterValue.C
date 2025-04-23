#include "TInterpreter.h"

extern "C" int printf(const char*,...);
struct WithDtor {
   WithDtor(): fBuf("IAmWithDtor!") {
      ++fgInstanceCount;
      printf("WithDtor(): %d\n", fgInstanceCount);
   }
   WithDtor(const WithDtor&) {
      ++fgInstanceCount;
      printf("WithDtor(const WithDtor&) %d\n",
             fgInstanceCount);
   }
   ~WithDtor() {
      --fgInstanceCount;
      printf("~WithDtor() %d\n", fgInstanceCount);
   }
   const char* ident() { return fBuf; }
   const char* fBuf;
   static int fgInstanceCount;
};

int WithDtor::fgInstanceCount = 0;

WithDtor returnsWithDtor() {
   WithDtor obj;
   printf("About to return a WithDtor\n");
   return obj;
}

void runInterpreterValue() {
   ClassInfo_t* ci = gInterpreter->ClassInfo_Factory("");
   CallFunc_t* cf = gInterpreter->CallFunc_Factory();
   gInterpreter->CallFunc_SetFunc(cf, ci, "returnsWithDtor", "", 0);
   
   TInterpreterValue* val = gInterpreter->CreateTemporary();
   gInterpreter->CallFunc_Exec(cf, 0, *val);

   WithDtor* withDtor = (WithDtor*)val->GetAsPointer();
   printf("Ident: %s\nNow deleting TInterpreterValue\n",
          withDtor->ident());
   delete val;
   printf("Now all WithDor should be gone; we have %d left\n",
          WithDtor::fgInstanceCount);
}
