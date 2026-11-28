// TEMP-DIAG: r2: MSVC and cling disagree on the layout of ROOT::Math::GradFunctor1D (vtordisp).
// 1. Probe classes: sizeof as compiled here versus sizeof in the interpreter.
// 2. All classes with a dictionary: sizeof from the dictionary versus sizeof in the interpreter.
// Fails on purpose on Windows, to get the output into the CI log.

#ifdef _WIN32

#include "TBaseClass.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TROOT.h"
#include "TSystem.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#define JC_NS JcCompiled
#include "JcTempVtordispProbes.h"

namespace {

void Print(const std::string &msg)
{
   fprintf(stderr, "[jc-TEMP] %s\n", msg.c_str());
   fflush(stderr);
}

bool HasVirtualBase(TClass *cl, int depth = 0)
{
   if (!cl || depth > 20 || !cl->HasInterpreterInfo())
      return false;
   TList *bases = cl->GetListOfBases();
   if (!bases)
      return false;
   for (auto obj : *bases) {
      auto base = static_cast<TBaseClass *>(obj);
      if (base->Property() & kIsVirtualBase)
         return true;
      if (HasVirtualBase(base->GetClassPointer(), depth + 1))
         return true;
   }
   return false;
}

void Probes()
{
   const unsigned n = JcCompiled::NProbes();
   const JcCompiled::Probe *compiled = JcCompiled::Probes();

   // the header is copied next to the test; else try the directory of this source file
   std::ifstream file("JcTempVtordispProbes.h");
   if (!file.is_open()) {
      std::string dir = __FILE__;
      dir.erase(dir.find_last_of("/\\") + 1);
      file.open(dir + "JcTempVtordispProbes.h");
   }
   const bool opened = file.is_open();
   std::stringstream text;
   text << "#define JC_NS JcCling\n";
   if (opened)
      text << file.rdbuf();
   const bool declared = opened && gInterpreter->Declare(text.str().c_str());
   Print(std::string("probes: ") + std::to_string(n) + ", header read: " + (opened ? "yes" : "NO") +
         ", declared in the interpreter: " + (declared ? "yes" : "NO") + ", sizeof(void*) = " +
         std::to_string(sizeof(void *)));

   const JcCompiled::Probe *cling = nullptr;
   if (declared && (unsigned)gInterpreter->Calc("(Longptr_t)JcCling::NProbes()") == n)
      cling = reinterpret_cast<const JcCompiled::Probe *>(gInterpreter->Calc("(Longptr_t)JcCling::Probes()"));

   unsigned nDiff = 0;
   for (unsigned i = 0; i < n; ++i) {
      char line[256];
      const int clingSize = cling ? (int)cling[i].fSize : -1;
      const bool diff = cling && cling[i].fSize != compiled[i].fSize;
      nDiff += diff;
      snprintf(line, sizeof(line), "probe %-30s compiled=%-3u cling=%-3d%s", compiled[i].fName, compiled[i].fSize,
               clingSize, diff ? " DIFF" : "");
      Print(line);
   }
   Print("probes with a different size: " + std::to_string(nDiff));
}

void Survey()
{
   const UInt_t nLibs = gSystem->LoadAllLibraries();
   gClassTable->Init();
   const Int_t nClasses = gClassTable->Classes();
   Print("survey: " + std::to_string(nLibs) + " libraries loaded, " + std::to_string(nClasses) + " classes");

   int nChecked = 0, nDiff = 0, nVBase = 0, nPrinted = 0;
   for (Int_t i = 0; i < nClasses; ++i) {
      const char *cname = gClassTable->Next();
      // std::tuple is described by a stand-in class
      if (!cname || !strncmp(cname, "tuple<", 6) || !strncmp(cname, "std::tuple<", 11))
         continue;
      TClass *cl = TClass::GetClass(cname, kTRUE);
      if (!cl || !cl->HasInterpreterInfo())
         continue;
      ClassInfo_t *info = cl->GetClassInfo();
      if (!info)
         continue;
      const int compiled = cl->Size();
      const int cling = gInterpreter->ClassInfo_Size(info);
      if (compiled <= 0 || cling <= 0)
         continue;
      ++nChecked;
      const bool vbase = HasVirtualBase(cl);
      const bool diff = compiled != cling;
      nDiff += diff;
      nVBase += vbase;
      if ((diff || vbase) && nPrinted < 500) {
         ++nPrinted;
         const Long_t prop = cl->ClassProperty();
         char line[512];
         snprintf(line, sizeof(line), "class %s: dictionary=%d cling=%d%s%s ctor=%s dtor=%s%s", cname, compiled, cling,
                  vbase ? " vbase" : "", diff ? " DIFF" : "", (prop & kClassHasExplicitCtor) ? "explicit" : "implicit",
                  (prop & kClassHasExplicitDtor) ? "explicit" : "implicit", (prop & kClassIsAbstract) ? " abstract" : "");
         Print(line);
      }
   }
   Print("survey: " + std::to_string(nChecked) + " classes compared, " + std::to_string(nVBase) +
         " with a virtual base, " + std::to_string(nDiff) + " with a different size");
}

} // namespace

int main()
{
   gROOT->SetBatch(kTRUE);
   Probes();
   Survey();
   return 1;
}

#else

int main()
{
   return 0;
}

#endif
