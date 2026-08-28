// Workspace-I/O tests for the JIT-free RooFormula evaluation backend:
// round-tripping formulas written through the AST path, reading a legacy
// workspace written by a pre-AST (TFormula-backed) build, and checking that
// both backends persist an identical on-disk representation.
// Author: Jonas Rembser, CERN 2026

#include "../src/RooFormulaUtils.h"

#include <RooAbsReal.h>
#include <RooCategory.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TFile.h>
#include <TStreamerInfo.h>
#include <TSystem.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>

namespace {

/// Bitwise comparison; any two NaNs count as equal.
bool sameBits(double a, double b)
{
   if (std::isnan(a) && std::isnan(b))
      return true;
   return std::memcmp(&a, &b, sizeof(double)) == 0;
}

/// Set or unset ROOFIT_FORMULA_BACKEND for the lifetime of the object,
/// resetting the backend's read-once cache on both ends.
class ScopedBackendEnv {
public:
   ScopedBackendEnv(const char *value)
   {
      if (const char *old = std::getenv("ROOFIT_FORMULA_BACKEND"))
         _old = old;
      // gSystem instead of setenv()/unsetenv(), which MSVC does not have
      if (value)
         gSystem->Setenv("ROOFIT_FORMULA_BACKEND", value);
      else
         gSystem->Unsetenv("ROOFIT_FORMULA_BACKEND");
      RooFormulaInternal::resetFormulaBackendForTesting();
   }
   ~ScopedBackendEnv()
   {
      if (_old.empty())
         gSystem->Unsetenv("ROOFIT_FORMULA_BACKEND");
      else
         gSystem->Setenv("ROOFIT_FORMULA_BACKEND", _old.c_str());
      RooFormulaInternal::resetFormulaBackendForTesting();
   }

private:
   std::string _old;
};

const char *const kObjectNames[] = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "p1"};

/// Fill the workspace with the same formula content that the legacy fixture
/// testRooFormulaEvaluator_legacy_ws.root was generated from (by a pre-AST,
/// Phase-1-only build whose persisted output is identical to unpatched ROOT):
/// plain named variables, @-references, TMath:: functions, `^`, a ternary,
/// and a category-state formula.
void fillWorkspace(RooWorkspace &ws)
{
   RooRealVar x("x", "x", 0.5, -10, 10);
   RooRealVar a("a", "a", 1.1, 0.1, 5);
   RooRealVar b("b", "b", 2.3, 0.1, 5);

   RooFormulaVar f1("f1", "simple", "a*x + b", {a, x, b});
   RooFormulaVar f2("f2", "atref", "@0*@1 - @2", {a, x, b});
   RooFormulaVar f3("f3", "funcs", "sqrt(abs(x)) + TMath::Erf(a) + exp(-b)", {x, a, b});
   RooFormulaVar f4("f4", "pow", "x^2 + a^-0.5 + pow(b, 3)", {x, a, b});
   RooFormulaVar f5("f5", "ternary", "x > 0 ? log(1+x) : -x", {x});
   RooFormulaVar f6("f6", "histfactory-shape", "1 + 0.02*a", {a});
   RooGenericPdf p1("p1", "genpdf", "exp(-0.5*((x-a)/b)^2)", {x, a, b});

   RooCategory c("c", "c");
   c.defineType("sig", 0);
   c.defineType("bkg", 1);
   RooFormulaVar f7("f7", "catstate", "c == c::sig ? a : b", {c, a, b});

   ws.import(f1);
   ws.import(f2);
   ws.import(f3);
   ws.import(f4);
   ws.import(f5);
   ws.import(f6);
   ws.import(p1);
   ws.import(f7);
}

/// The persisted formula string (_formExpr) of a RooFormulaVar or
/// RooGenericPdf, read without constructing the evaluation engine.
std::string persistedExpression(RooAbsArg *arg)
{
   if (auto *formulaVar = dynamic_cast<RooFormulaVar *>(arg)) {
      return formulaVar->expression();
   }
   if (auto *genericPdf = dynamic_cast<RooGenericPdf *>(arg)) {
      return genericPdf->expression();
   }
   return {};
}

/// Map of class name -> streamed class version for all TStreamerInfos in the
/// given file.
std::map<std::string, int> streamerInfoVersions(TFile &file)
{
   std::map<std::string, int> out;
   std::unique_ptr<TList> infos{file.GetStreamerInfoList()};
   for (TObject *obj : *infos) {
      if (auto *info = dynamic_cast<TStreamerInfo *>(obj)) {
         out[info->GetName()] = info->GetClassVersion();
      }
   }
   return out;
}

} // namespace

// Round-trip regression test for the formulaString() persistence landmine:
// RooFormulaVar/RooGenericPdf must persist a non-empty processed formula
// string when the AST backend is active (no TFormula exists to read a title
// from), and evaluate bitwise identically after reading back.
TEST(RooFormulaEvaluatorIO, WorkspaceRoundTrip)
{
   // Force the AST path: with a silent fallback this test could pass without
   // testing anything, and a formula that stops parsing shows up as a throw.
   ScopedBackendEnv env{"ast"};

   const char *fileName = "testRooFormulaEvaluatorIO_roundtrip.root";

   std::map<std::string, double> refValues;
   std::map<std::string, std::string> refExprs;
   {
      RooWorkspace ws{"w"};
      fillWorkspace(ws);
      for (const char *name : kObjectNames) {
         auto *arg = dynamic_cast<RooAbsReal *>(ws.arg(name));
         ASSERT_NE(arg, nullptr) << name;
         refValues[name] = arg->getVal();
         // after construction, _formExpr holds the processed x[i]-dialect
         // string; it must not be empty
         refExprs[name] = persistedExpression(arg);
         ASSERT_FALSE(refExprs[name].empty()) << name;
      }
      ASSERT_TRUE(ws.writeToFile(fileName)); // returns true on success
   }

   {
      TFile file(fileName, "READ");
      ASSERT_FALSE(file.IsZombie());
      auto *ws = file.Get<RooWorkspace>("w");
      ASSERT_NE(ws, nullptr);
      for (const char *name : kObjectNames) {
         auto *arg = dynamic_cast<RooAbsReal *>(ws->arg(name));
         ASSERT_NE(arg, nullptr) << name;
         EXPECT_EQ(persistedExpression(arg), refExprs[name]) << name;
         EXPECT_TRUE(sameBits(arg->getVal(), refValues[name]))
            << name << ": pre-write = " << std::hexfloat << refValues[name] << " read-back = " << arg->getVal()
            << std::defaultfloat;
      }
   }

   gSystem->Unlink(fileName);
}

namespace {

// Reference values recorded (with 17 significant digits, so the double
// literals below reproduce them bitwise) by the pre-AST build that wrote
// testRooFormulaEvaluator_legacy_ws.root. That build evaluated through
// TFormula/cling, so these are the historical values that reading the file
// must reproduce exactly.
const std::pair<const char *, double> kLegacyReference[] = {
   {"f1", 2.8499999999999996},  // a*x + b
   {"f2", -1.7499999999999998}, // @0*@1 - @2
   {"f3", 1.687570694483433},   // sqrt(abs(x)) + TMath::Erf(a) + exp(-b)
   {"f4", 13.370462589245591},  // x^2 + a^-0.5 + pow(b, 3)
   {"f5", 0.40546510810816438}, // x > 0 ? log(1+x) : -x
   {"f6", 1.022},               // 1 + 0.02*a
   {"f7", 1.1000000000000001},  // c == c::sig ? a : b
   {"p1", 0.96654592463371813}, // exp(-0.5*((x-a)/b)^2)
};

} // namespace

// Reading a workspace written by a pre-AST build must reproduce the recorded
// values bitwise, on the AST path (which all persisted strings must reach:
// they are stored in the processed x[i] dialect the parser targets) as well
// as on the TFormula fallback path.
TEST(RooFormulaEvaluatorIO, LegacyWorkspaceValues)
{
   for (const char *backend : {"ast", "tformula"}) {
      ScopedBackendEnv env{backend};
      TFile file("testRooFormulaEvaluator_legacy_ws.root", "READ");
      ASSERT_FALSE(file.IsZombie());
      auto *ws = file.Get<RooWorkspace>("w");
      ASSERT_NE(ws, nullptr);
      for (auto const &ref : kLegacyReference) {
         auto *arg = dynamic_cast<RooAbsReal *>(ws->arg(ref.first));
         ASSERT_NE(arg, nullptr) << ref.first;
         const double val = arg->getVal();
         EXPECT_TRUE(sameBits(val, ref.second))
            << ref.first << " (backend " << backend << "): expected = " << std::hexfloat << ref.second
            << " actual = " << val << std::defaultfloat;
      }
   }
}

// Fidelity of the persisted form: writing the same workspace content through
// the AST backend and through the TFormula backend must produce identical
// persisted formula strings and identical streamed class versions -- i.e. the
// on-disk representation does not depend on the evaluation backend, and it
// matches what the pre-AST build wrote. (A true cross-version read test with
// an unpatched ROOT release is complementary to this in-process check.)
TEST(RooFormulaEvaluatorIO, BackendWriteFidelity)
{
   const char *fileNameAst = "testRooFormulaEvaluatorIO_fidelity_ast.root";
   const char *fileNameTF = "testRooFormulaEvaluatorIO_fidelity_tformula.root";

   {
      ScopedBackendEnv env{"ast"};
      RooWorkspace ws{"w"};
      fillWorkspace(ws);
      ASSERT_TRUE(ws.writeToFile(fileNameAst));
   }
   {
      ScopedBackendEnv env{"tformula"};
      RooWorkspace ws{"w"};
      fillWorkspace(ws);
      ASSERT_TRUE(ws.writeToFile(fileNameTF));
   }

   TFile fileAst(fileNameAst, "READ");
   TFile fileTF(fileNameTF, "READ");
   TFile fileLegacy("testRooFormulaEvaluator_legacy_ws.root", "READ");
   ASSERT_FALSE(fileAst.IsZombie());
   ASSERT_FALSE(fileTF.IsZombie());
   ASSERT_FALSE(fileLegacy.IsZombie());

   auto *wsAst = fileAst.Get<RooWorkspace>("w");
   auto *wsTF = fileTF.Get<RooWorkspace>("w");
   auto *wsLegacy = fileLegacy.Get<RooWorkspace>("w");
   ASSERT_NE(wsAst, nullptr);
   ASSERT_NE(wsTF, nullptr);
   ASSERT_NE(wsLegacy, nullptr);

   for (const char *name : kObjectNames) {
      auto *argAst = dynamic_cast<RooAbsReal *>(wsAst->arg(name));
      auto *argTF = dynamic_cast<RooAbsReal *>(wsTF->arg(name));
      auto *argLegacy = dynamic_cast<RooAbsReal *>(wsLegacy->arg(name));
      ASSERT_NE(argAst, nullptr) << name;
      ASSERT_NE(argTF, nullptr) << name;
      ASSERT_NE(argLegacy, nullptr) << name;

      const std::string exprAst = persistedExpression(argAst);
      EXPECT_FALSE(exprAst.empty()) << name;
      EXPECT_EQ(exprAst, persistedExpression(argTF)) << name;
      EXPECT_EQ(exprAst, persistedExpression(argLegacy)) << name;

      EXPECT_TRUE(sameBits(argAst->getVal(), argTF->getVal())) << name;
      EXPECT_TRUE(sameBits(argAst->getVal(), argLegacy->getVal())) << name;
   }

   // The streamed class versions must be independent of the backend (no
   // schema change of any kind)...
   const auto versionsAst = streamerInfoVersions(fileAst);
   const auto versionsTF = streamerInfoVersions(fileTF);
   const auto versionsLegacy = streamerInfoVersions(fileLegacy);
   EXPECT_FALSE(versionsAst.empty());
   EXPECT_EQ(versionsAst, versionsTF);
   // ... and every class streamed by the current code must keep the version
   // it had in the legacy file. (The legacy file additionally contains a
   // pair<string,string> dictionary entry from the "origName" string
   // attributes that the RooFormula-based code stamped on imported formula
   // dependents; the current code doesn't write these noise attributes.)
   for (auto const &entry : versionsAst) {
      auto it = versionsLegacy.find(entry.first);
      ASSERT_NE(it, versionsLegacy.end()) << entry.first;
      EXPECT_EQ(it->second, entry.second) << entry.first;
   }

   gSystem->Unlink(fileNameAst);
   gSystem->Unlink(fileNameTF);
}
