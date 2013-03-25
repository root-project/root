////////////////////////////////////////////////////////////////////////////////
// This test case assures that TCallFunc works used through the               //
// TInterpreter's interfaces. It tests very simple cases of functions on the  //
// global scope, which take few argument and return.                          //
////////////////////////////////////////////////////////////////////////////////
// Author: Vassil Vassilev

extern "C" int printf(const char* fmt, ...);

// Examples for checking the return type
void VoidFuncNoArgs() {
   printf("void VoidFuncNoArgs().\n");
}

int IntFuncNoArgs() {
   printf("int IntFuncNoArgs().\n");
   return -1;
}

double DoubleFuncNoArgs() {
   printf("double DoubleFuncNoArgs().\n");
   return -1.0;
}

Int_t IntTFuncNoArgs() {
   printf("Int_t IntTFuncNoArgs().\n");
   return (Int_t)0;
}

class MyClassReturn {
public:
   int a;
   MyClassReturn() : a(1) {}
   ~MyClassReturn() { a = 0; }
   void Print() { printf("a=%d\n", a); }
};
MyClassReturn* MyClassReturnNoArgs() {
   printf("MyClassReturn* MyClassReturnNoArgs().\n");
   return new MyClassReturn();
}

// End Examples for checking the return type


// Examples for checking the arguments
float* FloatPtrOneArg(int p) {
   printf("FloatPtrOneArg(int p).\n");
   p++;
   printf("p=%d\n", p);
   return new float(p);
}

// Variadic arguments

#include <iostream>
#include <cstdarg>
// simple printf
void VariadicArguments(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
 
    while (*fmt != '\0') {
        if (*fmt == 'd') {
            int i = va_arg(args, int);
            std::cout << i << '\n';
        } else if (*fmt == 'c') {
            // note automatic conversion to integral type
            int c = va_arg(args, int);
            std::cout << static_cast<char>(c) << '\n';
        } else if (*fmt == 'f') {
            double d = va_arg(args, double);
            std::cout << d << '\n';
        }
        ++fmt;
    }
 
    va_end(args);
}
 
// End Examples for checking the arguments


#include "TClass.h"
#include "TInterpreter.h"

void runsimpleFunc() {
   runAllThroughTInterpreterInterfaces();
   printf("======================================================\n");
   runAllThroughTMethodCall();
}

void runAllThroughTInterpreterInterfaces() {
   printf("Running through TInterpreter public interfaces...\n");
   // FIXME: Somebody has to document that gInterpreter->ClassInfo_Factory("")
   // is the global namespace.
   ClassInfo_t* globalNamespace = gInterpreter->ClassInfo_Factory("");
   CallFunc_t* mc = gInterpreter->CallFunc_Factory();
   long offset = 0;

   // Run VoidFuncNoArgs
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "VoidFuncNoArgs", "", &offset);
   gInterpreter->CallFunc_Exec( mc, /* void* */0);

   // Run IntFuncNoArgs
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "IntFuncNoArgs", "", &offset);
   Long_t result_long = gInterpreter->CallFunc_ExecInt( mc, /* void* */0);
   printf("Result of IntFuncNoArgs = %ld\n", result_long);

   // Run IntTFuncNoArgs
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "IntTFuncNoArgs", "", &offset);
   result_long = gInterpreter->CallFunc_ExecInt( mc, /* void* */0);
   printf("Result of IntFuncNoArgs = %ld\n", result_long);

   // Run DoubleFuncNoArgs
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "DoubleFuncNoArgs", "", &offset);
   Double_t result_double = gInterpreter->CallFunc_ExecDouble( mc, /* void* */0);
   printf("Result of IntFuncNoArgs = %f\n", result_double);

   // Run MyClassReturnNoArgs (ptr return)
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "MyClassReturnNoArgs", "", &offset);
   result_long = gInterpreter->CallFunc_ExecInt( mc, /* void* */0);
   printf("Result of MyClassReturnNoArgs = ");
   reinterpret_cast<MyClassReturn*>(result_long)->Print();

   // Run FloatPtrOneArg (ptr return)
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "FloatPtrOneArg", "int", &offset);
   gInterpreter->CallFunc_SetArg(mc, (Long_t)1+1);
   result_long = gInterpreter->CallFunc_ExecInt( mc, /* void* */0);
   printf("Result of FloatPtrOneArg = %f\n", *reinterpret_cast<float*>(result_long));

   // Run VariadicArguments
   // FIXME: Dependent on cling/test/Lookup/variadicFunc.C
   // gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "VariadicArguments", "const char *, ...", &offset);
   // gInterpreter->CallFunc_SetArgs(mc, "\"dcf\",3, 'a', 1.999");
   // gInterpreter->CallFunc_Exec( mc, /* void* */0);


   // Cleanup
   gInterpreter->CallFunc_Delete(mc);
   gInterpreter->ClassInfo_Delete(globalNamespace);
}

#include "TMethodCall.h"

void runAllThroughTMethodCall() {
   printf("Running through TMethodCall...\n");
   TMethodCall method;
   Long_t result_long = 0;
   Double_t result_double;

   // Run VoidFuncNoArgs
   method = TMethodCall("VoidFuncNoArgs", "");
   method.Execute();

   // Run IntFuncNoArgs
   method = TMethodCall("IntFuncNoArgs", "");
   method.Execute(result_long);
   printf("Result of IntFuncNoArgs = %ld\n", result_long);

   // Run IntTFuncNoArgs
   method = TMethodCall("IntTFuncNoArgs", "");
   method.Execute(result_long);
   printf("Result of IntFuncNoArgs = %ld\n", result_long);

   // Run DoubleFuncNoArgs
   method = TMethodCall("DoubleFuncNoArgs", "");
   method.Execute(result_double);
   printf("Result of IntFuncNoArgs = %f\n", result_double);

   // Run MyClassReturnNoArgs (ptr return)
   method = TMethodCall("MyClassReturnNoArgs", "");
   method.Execute(result_long);
   printf("Result of MyClassReturnNoArgs = ");
   reinterpret_cast<MyClassReturn*>(result_long)->Print();

   // Run FloatPtrOneArg (ptr return)
   method = TMethodCall("FloatPtrOneArg", "1+1");
   method.Execute(result_long);

   printf("Result of FloatPtrOneArg = %f\n", *reinterpret_cast<float*>(result_long));

   // Run VariadicArguments
   // FIXME: Dependent on cling/test/Lookup/variadicFunc.C
   // method = TMethodCall("VariadicArguments", "\"dcf\",3, 'a', 1.999");
   // method.Execute();

}
