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
   Klass() { fKlass = this; }
   ~Klass() { fKlass = 0; }
   Klass* get() const { return fKlass; }
   static const int first_klf = 30;
   static const int last_klf = 130;
   long f30(double d) const {return (long)(d + 30);}
   long f31(double d) const {return (long)(d + 31);}
   long f32(double d) const {return (long)(d + 32);}
   long f33(double d) const {return (long)(d + 33);}
   long f34(double d) const {return (long)(d + 34);}
   long f35(double d) const {return (long)(d + 35);}
   long f36(double d) const {return (long)(d + 36);}
   long f37(double d) const {return (long)(d + 37);}
   long f38(double d) const {return (long)(d + 38);}
   long f39(double d) const {return (long)(d + 39);}
   long f40(double d) const {return (long)(d + 40);}
   long f41(double d) const {return (long)(d + 41);}
   long f42(double d) const {return (long)(d + 42);}
   long f43(double d) const {return (long)(d + 43);}
   long f44(double d) const {return (long)(d + 44);}
   long f45(double d) const {return (long)(d + 45);}
   long f46(double d) const {return (long)(d + 46);}
   long f47(double d) const {return (long)(d + 47);}
   long f48(double d) const {return (long)(d + 48);}
   long f49(double d) const {return (long)(d + 49);}
   long f50(double d) const {return (long)(d + 50);}
   long f51(double d) const {return (long)(d + 51);}
   long f52(double d) const {return (long)(d + 52);}
   long f53(double d) const {return (long)(d + 53);}
   long f54(double d) const {return (long)(d + 54);}
   long f55(double d) const {return (long)(d + 55);}
   long f56(double d) const {return (long)(d + 56);}
   long f57(double d) const {return (long)(d + 57);}
   long f58(double d) const {return (long)(d + 58);}
   long f59(double d) const {return (long)(d + 59);}
   long f60(double d) const {return (long)(d + 60);}
   long f61(double d) const {return (long)(d + 61);}
   long f62(double d) const {return (long)(d + 62);}
   long f63(double d) const {return (long)(d + 63);}
   long f64(double d) const {return (long)(d + 64);}
   long f65(double d) const {return (long)(d + 65);}
   long f66(double d) const {return (long)(d + 66);}
   long f67(double d) const {return (long)(d + 67);}
   long f68(double d) const {return (long)(d + 68);}
   long f69(double d) const {return (long)(d + 69);}
   long f70(double d) const {return (long)(d + 70);}
   long f71(double d) const {return (long)(d + 71);}
   long f72(double d) const {return (long)(d + 72);}
   long f73(double d) const {return (long)(d + 73);}
   long f74(double d) const {return (long)(d + 74);}
   long f75(double d) const {return (long)(d + 75);}
   long f76(double d) const {return (long)(d + 76);}
   long f77(double d) const {return (long)(d + 77);}
   long f78(double d) const {return (long)(d + 78);}
   long f79(double d) const {return (long)(d + 79);}
   long f80(double d) const {return (long)(d + 80);}
   long f81(double d) const {return (long)(d + 81);}
   long f82(double d) const {return (long)(d + 82);}
   long f83(double d) const {return (long)(d + 83);}
   long f84(double d) const {return (long)(d + 84);}
   long f85(double d) const {return (long)(d + 85);}
   long f86(double d) const {return (long)(d + 86);}
   long f87(double d) const {return (long)(d + 87);}
   long f88(double d) const {return (long)(d + 88);}
   long f89(double d) const {return (long)(d + 89);}
   long f90(double d) const {return (long)(d + 90);}
   long f91(double d) const {return (long)(d + 91);}
   long f92(double d) const {return (long)(d + 92);}
   long f93(double d) const {return (long)(d + 93);}
   long f94(double d) const {return (long)(d + 94);}
   long f95(double d) const {return (long)(d + 95);}
   long f96(double d) const {return (long)(d + 96);}
   long f97(double d) const {return (long)(d + 97);}
   long f98(double d) const {return (long)(d + 98);}
   long f99(double d) const {return (long)(d + 99);}
   long f100(double d) const {return (long)(d + 100);}
   long f101(double d) const {return (long)(d + 101);}
   long f102(double d) const {return (long)(d + 102);}
   long f103(double d) const {return (long)(d + 103);}
   long f104(double d) const {return (long)(d + 104);}
   long f105(double d) const {return (long)(d + 105);}
   long f106(double d) const {return (long)(d + 106);}
   long f107(double d) const {return (long)(d + 107);}
   long f108(double d) const {return (long)(d + 108);}
   long f109(double d) const {return (long)(d + 109);}
   long f110(double d) const {return (long)(d + 110);}
   long f111(double d) const {return (long)(d + 111);}
   long f112(double d) const {return (long)(d + 112);}
   long f113(double d) const {return (long)(d + 113);}
   long f114(double d) const {return (long)(d + 114);}
   long f115(double d) const {return (long)(d + 115);}
   long f116(double d) const {return (long)(d + 116);}
   long f117(double d) const {return (long)(d + 117);}
   long f118(double d) const {return (long)(d + 118);}
   long f119(double d) const {return (long)(d + 119);}
   long f120(double d) const {return (long)(d + 120);}
   long f121(double d) const {return (long)(d + 121);}
   long f122(double d) const {return (long)(d + 122);}
   long f123(double d) const {return (long)(d + 123);}
   long f124(double d) const {return (long)(d + 124);}
   long f125(double d) const {return (long)(d + 125);}
   long f126(double d) const {return (long)(d + 126);}
   long f127(double d) const {return (long)(d + 127);}
   long f128(double d) const {return (long)(d + 128);}
   long f129(double d) const {return (long)(d + 129);}
   long f130(double d) const {return (long)(d + 130);}

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
   res[0] = res[1] = 0;

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
      TString cmd
         = TString::Format("#include <vector>\n"
                           "class MyClass;\n"
                           "typedef MyClass* Klass%d_t;\n"
                           "std::vector<Klass%d_t> v%d;\n"
                           "void stressInterpreter_tmp%d() {\n"
                           "   v%d.push_back((Klass%d_t)0x12);\n"
                           "   *((int*)0x%lx) = 17;}",
                           i, i, i, i, i, i, (unsigned long) &res);
      TString tmpfilename = TString::Format("stressInterpreter_tmp%d.C", i);
      {
         std::ofstream otmp(tmpfilename.Data());
         otmp << cmd << endl;
      }
      gInterpreter->ProcessLine(TString(".X ") + tmpfilename, &interpError);
#ifndef ClingWorkAroundDeletedSourceFile
      gSystem->Unlink(tmpfilename);
#endif
      if (interpError != TInterpreter::kNoError) {
         printf("InterpreterStress::stressSTLDict(): "
                "Interpreter error: %d processing file %s:\n%s\n",
                interpError, tmpfilename.Data(), cmd.Data());
      } else if (res != 17) {
         printf("InterpreterStress::stressSTLDict(): "
                "Error getting correct result (expected %d, got %d) "
                "while processing file %s:\n%s\n",
                17, res, tmpfilename.Data(), cmd.Data());
      }
      allres &= (interpError == TInterpreter::kNoError);
      allres &= (res == 17);
      if (!allres) {
         break;
      }
   }
#ifdef ClingWorkAroundDeletedSourceFile
   for (Int_t i = 1; i < fNtimes; ++i) {
      TString tmpfilename = TString::Format("stressInterpreter_tmp%d.C", i);
      gSystem->Unlink(tmpfilename);
   }
#endif
   return allres;
}

/////////////////////////////////////////////////////////////////
// Test reflection query, reflection-based function call
bool InterpreterStress::stressReflection() {

   // This is fast
   int ntimes = fNtimes * 800;

   TString macro(fBinary);
   macro += ".cxx";
   gInterpreter->LoadMacro(macro);

   int numfuncs = Klass::last_klf - Klass::first_klf + 1;
   bool success = true;
   for (Int_t i = 0; success && i < ntimes; ++i) {
      int funcnum = i % (Long64_t)(1.2 * numfuncs);
      TString fname = TString::Format("f%d", funcnum);
      ClassInfo_t* k = gInterpreter->ClassInfo_Factory("Klass");
      bool hasMethod = gInterpreter->ClassInfo_HasMethod(k, fname.Data());
      if (hasMethod != (funcnum >= Klass::first_klf && funcnum <= Klass::last_klf)) {
         std::cout << "Error: Can not find " << fname << " in Klass\n";
         success = false;
      }
      if (!hasMethod) {
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }

      CallFunc_t* mk = gInterpreter->CallFunc_Factory();
      Long_t offset = -1;
      gInterpreter->CallFunc_SetFuncProto(mk, k, fname, "double", &offset);
      if (!gInterpreter->CallFunc_IsValid(mk)) {
         std::cout << "Error: The CallFunc for Klass::" << fname << "(double) is invalid\n";
         success = false;
         gInterpreter->CallFunc_Delete(mk);
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }
      if (offset != 0) {
         std::cout << "Error: The offset for CallFunc for Klass::" << fname << " is not zero (" << offset << ")\n";
         success = false;
         gInterpreter->CallFunc_Delete(mk);
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }

      gInterpreter->CallFunc_SetArg(mk, -funcnum * 2 + 0.2);

      void* obj = gInterpreter->ClassInfo_New(k);
      if (!obj) {
         std::cout << "Error: ClasInfo::New for Klass failed.\n";
         success = false;
         gInterpreter->CallFunc_Delete(mk);
         gInterpreter->ClassInfo_Delete(k);
         continue;
      }

      long ret = gInterpreter->CallFunc_ExecInt(mk, obj);
      if (ret != (long) (funcnum + (-funcnum * 2 + 0.2))) {
         std::cout << "Error: Execution of Klass::" << fname << " failed (result = "
                   << ret << " rather than = " << (long) (funcnum + (-funcnum * 2 + 0.2)) << ")\n";
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
      for (unsigned int pattern = 0; pattern < 0xff; ++pattern) {
         for (unsigned int bit = 0; bit < sizeof(Long64_t)*7; ++bit) {
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
   for (unsigned int itest = 0; itest < fNames.size(); ++itest) {
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
   // Since this routine can be called (almost) directly from the command line and is used
   // in automated test, it must return 0 in case of success
   InterpreterStress stress(binary);
   return !stress.run(ntimes, runTests);
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
            for (unsigned int i = 0; i < tmp.fNames.size(); ++i) alltests += TString(" ") + tmp.fNames[i];
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

   gROOT->SetBatch();
   TApplication theApp("App", &argc, argv);

   TString exe(argv[0]);
   if (exe.EndsWith(".exe")) exe.Remove(exe.Length() - 4, 4);

   gBenchmark = new TBenchmark();

   if (runInterpreted) {
      TString cmd = TString::Format(".L %s.cxx", exe.Data());
      gInterpreter->ProcessLine(cmd);
      exe = gSystem->BaseName(exe);
      cmd = TString::Format("%s(%d, \"%s\", \"%s\")", exe.Data(), ntimes, runTests.Data(), exe.Data());
      if (0 != gInterpreter->ProcessLine(cmd)) return 1;
   } else {
      if (0 != stressInterpreter(ntimes, runTests, exe)) return 1;
   }
   return 0;
}
#endif
