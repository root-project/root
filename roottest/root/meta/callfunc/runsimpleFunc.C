////////////////////////////////////////////////////////////////////////////////
// This test case assures that TCallFunc works used through TInterpreter's    //
// and TMethodCall's interfaces.                                              //
//                                                                            //
// It tests very simple cases of functions on the global scope, which take    //
// few arguments and return. It covers functions in namespaces and nested     //
// namespaces.                                                                //
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

// Functions in namespaces

#include "Rtypes.h"
namespace A {
   Double32_t* Double32TPtrThreeArgs(int p, float f, double* d) {
      printf("Double32PtrOneArg(int p, float f, double* d).\n");
      return new Double32_t(p + f - *d);
   }
}

namespace A {
   namespace A1 {
      namespace A2 {
         int Num = 3;
         namespace A3 {
            int NestedNamespaceIntOneArg(int a) {
               return A2::Num + a;
            }
         }
      }
   }
}

// End Functions in namespaces


#include "TClass.h"
#include "TInterpreter.h"

void runAllThroughTInterpreterInterfaces() {
   printf("Running through TInterpreter public interfaces...\n");
   // FIXME: Somebody has to document that gInterpreter->ClassInfo_Factory("")
   // is the global namespace.
   ClassInfo_t* globalNamespace = gInterpreter->ClassInfo_Factory("");
   CallFunc_t* mc = gInterpreter->CallFunc_Factory();
   Longptr_t offset = 0;

   // Run VoidFuncNoArgs
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "VoidFuncNoArgs", "", &offset);
   gInterpreter->CallFunc_Exec(mc, /* void* */0);

   // Run IntFuncNoArgs
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "IntFuncNoArgs", "", &offset);
   Longptr_t result_long = gInterpreter->CallFunc_ExecInt(mc, /* void* */0);
   printf("Result of IntFuncNoArgs = %zd\n", (size_t)result_long);

   // Run IntTFuncNoArgs
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "IntTFuncNoArgs", "", &offset);
   result_long = gInterpreter->CallFunc_ExecInt(mc, /* void* */0);
   printf("Result of IntFuncNoArgs = %zd\n", (size_t)result_long);

   // Run DoubleFuncNoArgs
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "DoubleFuncNoArgs", "", &offset);
   Double_t result_double = gInterpreter->CallFunc_ExecDouble(mc, /* void* */0);
   printf("Result of IntFuncNoArgs = %f\n", result_double);

   // Run MyClassReturnNoArgs (ptr return)
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "MyClassReturnNoArgs", "", &offset);
   result_long = gInterpreter->CallFunc_ExecInt(mc, /* void* */0);
   printf("Result of MyClassReturnNoArgs = ");
   reinterpret_cast<MyClassReturn*>(result_long)->Print();

   // Run VariadicArguments
   // FIXME: Dependent on cling/test/Lookup/variadicFunc.C
   // gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "VariadicArguments", "const char *, ...", &offset);
   // gInterpreter->CallFunc_SetArgs(mc, "\"dcf\",3, 'a', 1.999");
   // gInterpreter->CallFunc_Exec(mc, /* void* */0);

   // Run FloatPtrOneArg (ptr return)
   // Find namespace A:
   gInterpreter->CallFunc_SetFuncProto(mc, globalNamespace, "FloatPtrOneArg", "int", &offset);
   gInterpreter->CallFunc_SetArg(mc, (Long_t)1+1);
   result_long = gInterpreter->CallFunc_ExecInt(mc, /* void* */0);
   printf("Result of FloatPtrOneArg = %f\n", *reinterpret_cast<float*>(result_long));

   // Run A::Double32TPtrThreeArgs (ptr return)
   ClassInfo_t* namespaceA = gInterpreter->ClassInfo_Factory("A");
   gInterpreter->CallFunc_SetFuncProto(mc, namespaceA, "Double32TPtrThreeArgs", "int, float, double *", &offset);
   gInterpreter->CallFunc_SetArg(mc, (Long_t)1+1);
   gInterpreter->CallFunc_SetArg(mc, (Double_t)1.-1);
   gInterpreter->CallFunc_SetArg(mc, (ULong64_t)new double(3.000));
   result_long = gInterpreter->CallFunc_ExecInt(mc, /* void* */0);
   printf("Result of A::Double32TPtrThreeArgs = %f\n", *reinterpret_cast<Double32_t*>(result_long));

   // Run A::A1::A2::A3::NestedNamespaceIntOneArg (int return)
   ClassInfo_t* namespaceA3 = gInterpreter->ClassInfo_Factory("A::A1::A2::A3");
   gInterpreter->CallFunc_SetFuncProto(mc, namespaceA3, "NestedNamespaceIntOneArg", "int", &offset);
   gInterpreter->CallFunc_SetArg(mc, (Long_t)11);
   result_long = gInterpreter->CallFunc_ExecInt(mc, /* void* */0);
   printf("Result of A::A1::A2::A3::NestedNamespaceIntOneArg = %zd\n", (size_t)result_long);

   // Cleanup
   gInterpreter->CallFunc_Delete(mc);
   gInterpreter->ClassInfo_Delete(globalNamespace);
}

#include "TMethodCall.h"

void runAllThroughTMethodCall() {
   printf("Running through TMethodCall...\n");
   TMethodCall method;
   Longptr_t result_long = 0;
   Double_t result_double;

   // Run VoidFuncNoArgs
   method = TMethodCall("VoidFuncNoArgs", "");
   method.Execute();

   // Run IntFuncNoArgs
   method = TMethodCall("IntFuncNoArgs", "");
   method.Execute(result_long);
   printf("Result of IntFuncNoArgs = %zd\n", (size_t)result_long);

   // Run IntTFuncNoArgs
   method = TMethodCall("IntTFuncNoArgs", "");
   method.Execute(result_long);
   printf("Result of IntFuncNoArgs = %zd\n", (size_t)result_long);

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

   // Run A::Double32TPtrThreeArgs (ptr return)
   method = TMethodCall("A::Double32TPtrThreeArgs", "1+1, 1.-1, new double(3.000)");
   method.Execute(result_long);
   printf("Result of A::Double32TPtrThreeArgs = %f\n", *reinterpret_cast<Double32_t*>(result_long));

   // Run A::A1::A2::A3::NestedNamespaceIntOneArg (int return)
   method.InitWithPrototype("A::A1::A2::A3::NestedNamespaceIntOneArg", "int");
   method.SetParam((Long_t)11);
   method.Execute(result_long);
   printf("Result of A::A1::A2::A3::NestedNamespaceIntOneArg = %zd\n", (size_t)result_long);
}

void runsimpleFunc() {
   runAllThroughTInterpreterInterfaces();
   printf("======================================================\n");
   runAllThroughTMethodCall();
}
