// @(#)root/test:$Id$
// Author: Axel Naumann, 2011-01-11

/////////////////////////////////////////////////////////////////
//
// Stress test (functionality and timing) for C++ interpreter.
//
/////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "TSystem.h"
#include "TBenchmark.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TApplication.h"

/////////////////////////////////////////////////////////////////
// Utility classes / functions

class Base {};

class Klass: public Base {
public:
   Klass(): fKlass(this) {}
   ~Klass() { fKlass = 0; }
   Klass* get() const { return fKlass; }
   static const int first_klf = 30;
   static const int last_klf = 130;
   int f30(double d) const {return d + 30;}
   int f31(double d) const {return d + 31;}
   int f32(double d) const {return d + 32;}
   int f33(double d) const {return d + 33;}
   int f34(double d) const {return d + 34;}
   int f35(double d) const {return d + 35;}
   int f36(double d) const {return d + 36;}
   int f37(double d) const {return d + 37;}
   int f38(double d) const {return d + 38;}
   int f39(double d) const {return d + 39;}
   int f40(double d) const {return d + 40;}
   int f41(double d) const {return d + 41;}
   int f42(double d) const {return d + 42;}
   int f43(double d) const {return d + 43;}
   int f44(double d) const {return d + 44;}
   int f45(double d) const {return d + 45;}
   int f46(double d) const {return d + 46;}
   int f47(double d) const {return d + 47;}
   int f48(double d) const {return d + 48;}
   int f49(double d) const {return d + 49;}
   int f50(double d) const {return d + 50;}
   int f51(double d) const {return d + 51;}
   int f52(double d) const {return d + 52;}
   int f53(double d) const {return d + 53;}
   int f54(double d) const {return d + 54;}
   int f55(double d) const {return d + 55;}
   int f56(double d) const {return d + 56;}
   int f57(double d) const {return d + 57;}
   int f58(double d) const {return d + 58;}
   int f59(double d) const {return d + 59;}
   int f60(double d) const {return d + 60;}
   int f61(double d) const {return d + 61;}
   int f62(double d) const {return d + 62;}
   int f63(double d) const {return d + 63;}
   int f64(double d) const {return d + 64;}
   int f65(double d) const {return d + 65;}
   int f66(double d) const {return d + 66;}
   int f67(double d) const {return d + 67;}
   int f68(double d) const {return d + 68;}
   int f69(double d) const {return d + 69;}
   int f70(double d) const {return d + 70;}
   int f71(double d) const {return d + 71;}
   int f72(double d) const {return d + 72;}
   int f73(double d) const {return d + 73;}
   int f74(double d) const {return d + 74;}
   int f75(double d) const {return d + 75;}
   int f76(double d) const {return d + 76;}
   int f77(double d) const {return d + 77;}
   int f78(double d) const {return d + 78;}
   int f79(double d) const {return d + 79;}
   int f80(double d) const {return d + 80;}
   int f81(double d) const {return d + 81;}
   int f82(double d) const {return d + 82;}
   int f83(double d) const {return d + 83;}
   int f84(double d) const {return d + 84;}
   int f85(double d) const {return d + 85;}
   int f86(double d) const {return d + 86;}
   int f87(double d) const {return d + 87;}
   int f88(double d) const {return d + 88;}
   int f89(double d) const {return d + 89;}
   int f90(double d) const {return d + 90;}
   int f91(double d) const {return d + 91;}
   int f92(double d) const {return d + 92;}
   int f93(double d) const {return d + 93;}
   int f94(double d) const {return d + 94;}
   int f95(double d) const {return d + 95;}
   int f96(double d) const {return d + 96;}
   int f97(double d) const {return d + 97;}
   int f98(double d) const {return d + 98;}
   int f99(double d) const {return d + 99;}
   int f100(double d) const {return d + 100;}
   int f101(double d) const {return d + 101;}
   int f102(double d) const {return d + 102;}
   int f103(double d) const {return d + 103;}
   int f104(double d) const {return d + 104;}
   int f105(double d) const {return d + 105;}
   int f106(double d) const {return d + 106;}
   int f107(double d) const {return d + 107;}
   int f108(double d) const {return d + 108;}
   int f109(double d) const {return d + 109;}
   int f110(double d) const {return d + 110;}
   int f111(double d) const {return d + 111;}
   int f112(double d) const {return d + 112;}
   int f113(double d) const {return d + 113;}
   int f114(double d) const {return d + 114;}
   int f115(double d) const {return d + 115;}
   int f116(double d) const {return d + 116;}
   int f117(double d) const {return d + 117;}
   int f118(double d) const {return d + 118;}
   int f119(double d) const {return d + 119;}
   int f120(double d) const {return d + 120;}
   int f121(double d) const {return d + 121;}
   int f122(double d) const {return d + 122;}
   int f123(double d) const {return d + 123;}
   int f124(double d) const {return d + 124;}
   int f125(double d) const {return d + 125;}
   int f126(double d) const {return d + 126;}
   int f127(double d) const {return d + 127;}
   int f128(double d) const {return d + 128;}
   int f129(double d) const {return d + 129;}
   int f130(double d) const {return d + 130;}
   
private:
   Klass* fKlass;
};

unsigned long func(Long64_t& a, double b, const Klass& c) {
   if (--a > b) return func(a, b, c);
   return (unsigned long) c.get();
}

class InterpreterStress {
public:
   InterpreterStress(const char* binary): fNtimes(10), fBinary(binary) {
      fNames.push_back("FuncCall");
      fNames.push_back("STLDict");
      fNames.push_back("Reflection");
      fNames.push_back("NestedStatements");
   }

   bool run(Int_t ntimes = 10, const char* runTests = 0);

   bool stressFuncCall();

   void prepareSTLDict();
   bool stressSTLDict();

   bool stressReflection();

   bool stressNestedStatements();

   std::vector<std::string> fNames;

private:
   void runPreps() {
      prepareSTLDict();
   }

   Int_t fNtimes;
   TString fBinary;
};

/////////////////////////////////////////////////////////////////
// Test function call performance

bool InterpreterStress::stressFuncCall() {

   // This is fast.
   int ntimes = fNtimes * 100000;

   Klass c;
   unsigned long res[2];
   int depth = 6; // That's all that Windows can handle...
   for (int i = 0; i < ntimes / depth; ++i) {
      // Call recursively:
      Long64_t a = depth;
      res[0] = func(a, 0., c);
   }

   // Call non-recursively:
   for (Long64_t a = ntimes; a > 0;) {
      res[1] = func(a, a - 1, c);
   }
   if (res[0] != (unsigned long)&c) return false;
   if (res[0] != res[1]) return false;
   return true;
}


/////////////////////////////////////////////////////////////////
// Test custom STL dictionary / calls
void InterpreterStress::prepareSTLDict() {
   // Remove AutoDict
   void* dir = gSystem->OpenDirectory(gSystem->pwd());
   const char* name = 0;
   while ((name = gSystem->GetDirEntry(dir))) {
      if (!strncmp(name, "AutoDict_", 9)) {
         gSystem->Unlink(name);
      }
   }
   gSystem->FreeDirectory(dir);
}
bool InterpreterStress::stressSTLDict() {
   using namespace std;

   bool allres = true;
   for (Int_t i = 1; i < fNtimes; ++i) {
      int res = 3;
      TInterpreter::EErrorCode interpError = TInterpreter::kNoError;
      TString cmd = TString::Format("#include <vector>\nclass MyClass;\ntypedef MyClass* Klass%d_t;\nstd::vector<Klass%d_t> v%d;\nvoid stressInterpreter_tmp%d() {\n   v%d.push_back((Klass%d_t)0x12);\n   *((int*)0x%lx) = 0;}", i, i, i, i, i, i, (unsigned long) &res);
      TString tmpfilename = TString::Format("stressInterpreter_tmp%d.C", i);
      {
         std::ofstream otmp(tmpfilename.Data());
         otmp << cmd << endl;
      }
      gInterpreter->ProcessLine(TString(".X ") + tmpfilename, &interpError);
      gSystem->Unlink(tmpfilename);
      allres &= (interpError == TInterpreter::kNoError);
      allres &= (res == 0);
   }
   return allres;
}

/////////////////////////////////////////////////////////////////
// Test reflection query, reflection-based function call
bool InterpreterStress::stressReflection() {

   // This is fast
   int ntimes = fNtimes * 800;

#if !defined(__CINT__) && !defined(__CLING__)
   TString macro(fBinary);
   macro += ".cxx";
   gInterpreter->LoadMacro(macro);
#endif
   int numfuncs = Klass::last_klf - Klass::first_klf + 1;
   bool success = true;
   for (Int_t i = 0; i < ntimes; ++i) {
      int funcnum = i % (Long64_t)(1.2 * numfuncs);
      TString fname = TString::Format("f%d", funcnum);
      ClassInfo_t* k = gInterpreter->ClassInfo_Factory("Klass");
      bool hasMethod = gInterpreter->ClassInfo_HasMethod(k, fname.Data());
      if (hasMethod != (funcnum >= Klass::first_klf && funcnum <= Klass::last_klf)) {
         success = false;
      }
      if (!hasMethod) {
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }

      MethodInfo_t* mk = gInterpreter->CallFunc_Factory();
      Long_t offset = -1;
      gInterpreter->CallFunc_SetFuncProto(mk, k, fname, "double", &offset);
      if (!gInterpreter->CallFunc_IsValid(mk)) {
         success = false;
         gInterpreter->CallFunc_Delete(mk);
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }
      if (offset != 0) {
         success = false;
         gInterpreter->CallFunc_Delete(mk);
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }

      gInterpreter->CallFunc_SetArg(mk, -funcnum * 2 + 0.2);

      void* obj = gInterpreter->ClassInfo_New(k);
      if (!obj) {
         success = false;
         gInterpreter->CallFunc_Delete(mk);
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }

      double ret = gInterpreter->CallFunc_ExecDouble(mk, obj);
      if (ret != funcnum + (-funcnum * 2 + 0.2)) {
         success = false;
         gInterpreter->CallFunc_Delete(mk);
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }

      gInterpreter->ClassInfo_Delete(k, obj);

      gInterpreter->CallFunc_Delete(mk);
      gInterpreter->ClassInfo_Delete(k);
   }

   return success;
}

/////////////////////////////////////////////////////////////////
// Test nested compound statements (if, switch, for,...)
bool InterpreterStress::stressNestedStatements() {
   bool success = true;
   int ntimes = fNtimes * 4;
   for (int i = 0; i < ntimes; ++i) {
      for (int pattern = 0; pattern < 0xff; ++pattern) {
         for (int bit = 0; bit < sizeof(Long64_t)*7; ++bit) {
            ULong64_t v = 1; // always > 0
            switch (pattern & 0xf) {
            case 1: v += 1;
            case 3: v += 3;
            case 5: v += 5;
            case 7: v += 7;
            case 9: v += 9;
            case 11: v += 11;
            case 13: v += 13;
            case 15: v += 15;
            default:
               v += (pattern & 0xf);
            }
            v = v << bit;

            if (bit < 32) {
               if (v > (1ll << 48)) success = false;
            } else {
               if (pattern) {
                  if (!v || v < 1) success = false;
                  else {
                     if (bit > 0 && v == pattern && pattern > 0) {
                        if (success)
                           success = false;
                     }
                  }
               } else {
                  if (success) success = true;
               }
            }
            while (v) v = v >> 1;
            if (v) success = false;
         }
      }
   }
   return success;
}

/////////////////////////////////////////////////////////////////
// Driver

bool InterpreterStress::run(Int_t ntimes /*= 10*/, const char* runTests /*= 0*/) {
   using namespace std;
   static const char* benchmark = "stressInterpreter";

   fNtimes = ntimes;

   runPreps();

   gBenchmark->Start(benchmark);
   cout << "****************************************************************************" <<endl;
   cout << "*  Starting  stress INTERPRETER                                            *" <<endl;
   cout << "****************************************************************************" <<endl;
   bool success = true;
   for (int itest = 0; itest < fNames.size(); ++itest) {
      if (runTests && runTests[0]) {
         // only run test if it was selected
         if (!strstr(fNames[itest].c_str(), runTests)) continue;
      }
      bool res = false;
      switch (itest) {
      case 0: res = stressFuncCall(); break;
      case 1: res = stressSTLDict(); break;
      case 2: res = stressReflection(); break;
      case 3: res = stressNestedStatements(); break;
      }
      success &= res;
      printf("%s %s%s\n", fNames[itest].c_str(), TString('.', 77 - fNames[itest].length() - 8).Data(), (res ? "... OK" : " FAILED"));
   }

   // Summary:
   gBenchmark->Stop(benchmark);
   Double_t reftime100 = 600; //pcbrun compiled
   Double_t ct = gBenchmark->GetCpuTime(benchmark);
   const Double_t rootmarks = 800*reftime100*ntimes/(100*ct);
   printf("****************************************************************************\n");

   gBenchmark->Print(benchmark);
   printf("****************************************************************************\n");
   printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),
         gROOT->GetVersionDate(),gROOT->GetVersionTime());
   printf("****************************************************************************\n");

   return success;
}

bool stressInterpreter(Int_t ntimes = 10, const char* runTests = 0, const char* binary = "") {
   InterpreterStress stress(binary);
   return stress.run(ntimes, runTests);
}

#if !defined(__CINT__) && !defined(__CLING__)
// If compiled: interpret! (by default)

int main(int argc, char **argv)
{
   Int_t ntimes = 2;
   bool runInterpreted = true;
   TString runTests;

   for (int iarg = 1; iarg < argc; ++iarg) {
      const char* arg = argv[iarg];
      if (arg[0] == '-') {
         if (!strcmp(arg, "--help")) {
            printf("Interpreter speed test\n");
            printf("Run as: %s [--help] [-c] [--test=...] [num]\n", gSystem->BaseName(argv[0]));
            printf("  --help: print this help\n");
            printf("  -c: run compiled\n");
            TString alltests;
            InterpreterStress tmp("");
            for (int i = 0; i < tmp.fNames.size(); ++i) alltests += TString(" ") + tmp.fNames[i];
            printf("  --test=...: run only given test, one of%s\n", alltests.Data());
            printf("  num: run for num iterations (default: %d)\n", ntimes);
            return 0;
         }
         if (!strcmp(arg, "-c")) {
            runInterpreted = false;
            continue;
         }
         if (!strncmp(arg, "--test=", 7)) {
            runTests = arg + 7;
            continue;
         }
      } else {
         ntimes = atoi(argv[1]);
      }
   }

   TApplication theApp("App", &argc, argv);
   gROOT->SetBatch();

   TString exe(argv[0]);
   if (exe.EndsWith(".exe")) exe.Remove(exe.Length() - 4, 4);

   gBenchmark = new TBenchmark();

   if (runInterpreted) {
      TString cmd = TString::Format(".L %s.cxx", exe.Data());
      gInterpreter->ProcessLine(cmd);
      exe = gSystem->BaseName(exe);
      cmd = TString::Format("%s(%d, \"%s\", \"%s\")", exe.Data(), ntimes, runTests.Data(), exe.Data());
      if (!gInterpreter->ProcessLine(cmd)) return 1;
   } else {
      if (!stressInterpreter(ntimes, runTests, exe)) return 1;
   }
   return 0;
}
#endif
