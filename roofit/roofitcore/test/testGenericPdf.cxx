// Tests for the GenericPdf
// Authors: Stephan Hageboeck, CERN  05/2019
//          Jonas Rembser, CERN 06/2022
#include <TFile.h>

#include <RooArgList.h>
#include <RooBinning.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooHistPdf.h>
#include <RooProduct.h>
#include <RooRealVar.h>
#include <RooUniformBinning.h>
#include <RooWorkspace.h>

#include <gtest/gtest.h>

#include <memory>

#define MAKE_JOHNSON_AND_VARS RooRealVar mass("mass", "mass", 0., -200., 200.);\
RooRealVar mu("mu", "Location parameter of normal distribution", 100., -200., 200.);\
RooRealVar sigma("sigma", "Two sigma of normal distribution", 2., 0., 100.);\
RooRealVar gamma("gamma", "gamma", -10., -100., 100.);\
RooRealVar delta("delta", "delta", 3., 0., 100.);

const char* fixedFormula = "delta/(sigma*std::sqrt(TMath::TwoPi()))"
    "*std::exp(-0.5*(gamma+delta*TMath::ASinH((mass-mu)/sigma))"
                    "*(gamma+delta*TMath::ASinH((mass-mu)/sigma)))"
    "/std::sqrt(1+(mass-mu)*(mass-mu)/(sigma*sigma))";



TEST(GenericPdf, CrashWhenRunningJohnson)
{
  MAKE_JOHNSON_AND_VARS
  RooGenericPdf johnsonRef("johnsonRef",
      fixedFormula,
      RooArgSet( mass, mu, sigma, gamma, delta));

  for (double theMass : {1., 10., 50., 100.}) {
    mass = theMass;
    EXPECT_NE(johnsonRef.getVal(), 0.) << theMass;
  }
}

// ROOT-10411
TEST(GenericPdf, CrashWhenRenamingArguments)
{
  RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

  RooRealVar var("var", "var", 0.1, 0, 1);
  RooRealVar par("par", "par", 0.5, 0, 1);
  RooGenericPdf genPdf("genPdf", "var*(par + 1)", RooArgSet(var, par));

  par.SetName("par_test");

  auto pdfClone = (RooAbsPdf*)genPdf.clone("clone");
  // Would crash:
  EXPECT_NEAR(pdfClone->getVal(), 0.15, 1.E-6);

  par.SetName("par");

  RooWorkspace ws("ws");
  ws.import(genPdf, RooFit::RenameAllVariablesExcept("new", "var"));
  auto impPdf = ws.pdf("genPdf");
  // Would crash:
  EXPECT_NEAR(impPdf->getVal(), 0.15, 1.E-6);
}

// ROOT-5101: Identity PDF affects normalization
TEST(GenericPdf, IdentidyPdfNormalization)
{
  RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

  RooWorkspace ws;
  ws.factory("Exponential::exp(x[0.0, 100.0], s[-0.5, -10.0,  0.0])");

  RooRealVar& x = *ws.var("x");
  RooAbsPdf& exp = *ws.pdf("exp");

  // This RooGenericPdf should behave exactly the same as the exponential, only
  // that the integration will be done numerically.
  RooGenericPdf pdf{"pdf", "pdf", "@0", {exp}};

  RooArgSet normSet{x};

  // Check that the values with and without normalization are almost identical.
  // They are not exactly identical for the normalized case, because the
  // RooGenericPdf doesn't do analytic integration.
  constexpr double tol = 1e-6;
  EXPECT_NEAR(exp.getVal(), pdf.getVal(), tol);
  EXPECT_NEAR(exp.getVal(normSet), pdf.getVal(normSet), tol);
}

// Tests for RooGenericPdf::setBinning(), which declares the formula to be
// piecewise constant (flat) within bins of an observable so that integration
// can use the fast bin integrator instead of the generic numeric integrator.

// A pdf that is flat within five uniform bins on [0, 10], with one height per bin.
#define MAKE_PIECEWISE_FLAT_VARS    \
   RooRealVar x("x", "x", 0., 10.); \
   RooRealVar h0("h0", "", 1.0), h1("h1", "", 3.0), h2("h2", "", 2.0), h3("h3", "", 4.0), h4("h4", "", 1.5);

const char *piecewiseFlatFormula = "(floor(x/2)==0)*h0+(floor(x/2)==1)*h1+(floor(x/2)==2)*h2"
                                   "+(floor(x/2)==3)*h3+(floor(x/2)==4)*h4";

// Exact integral of the piecewise-flat function: sum(height) * binWidth.
constexpr double piecewiseFlatIntegral = (1.0 + 3.0 + 2.0 + 4.0 + 1.5) * 2.0; // = 23

namespace {

// Integrate `pdf` over `obs`, returning the integral value and reporting the
// name of the numeric integrator that was used via `integratorName`.
double integrate(RooAbsReal &pdf, const RooArgSet &obs, std::string &integratorName)
{
   RooHelpers::HijackMessageStream hijack(RooFit::INFO, RooFit::NumericIntegration);
   std::unique_ptr<RooAbsReal> integ{pdf.createIntegral(obs)};
   const double value = integ->getVal();
   integratorName = hijack.str();
   return value;
}

} // namespace

TEST(GenericPdf, BinnedIntegrationExplicitBoundaries)
{
   MAKE_PIECEWISE_FLAT_VARS
   RooGenericPdf pdf("pdf", "", piecewiseFlatFormula, RooArgList(x, h0, h1, h2, h3, h4));

   // Without a binning, the generic numeric integrator is used.
   {
      std::string integratorName;
      integrate(pdf, x, integratorName);
      EXPECT_EQ(integratorName.find("RooBinIntegrator"), std::string::npos) << integratorName;
   }

   const double boundaries[] = {0, 2, 4, 6, 8, 10};
   pdf.setBinning(x, RooBinning(5, boundaries));

   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(x)));

   std::string integratorName;
   const double value = integrate(pdf, x, integratorName);
   EXPECT_NE(integratorName.find("RooBinIntegrator"), std::string::npos) << integratorName;
   EXPECT_DOUBLE_EQ(value, piecewiseFlatIntegral);
}

// removeBinning() reverses setBinning(), reverting to the generic integrator.
TEST(GenericPdf, BinnedRemoveBinning)
{
   MAKE_PIECEWISE_FLAT_VARS
   RooGenericPdf pdf("pdf", "", piecewiseFlatFormula, RooArgList(x, h0, h1, h2, h3, h4));

   pdf.setBinning(x, RooUniformBinning(0.0, 10.0, 5));
   ASSERT_TRUE(pdf.isBinnedDistribution(RooArgSet(x)));
   ASSERT_NE(pdf.getBinning(x), nullptr);
   EXPECT_TRUE(pdf.getBinning(x)->isUniform());

   // Removing the binning returns true and reverts to the generic integrator.
   EXPECT_TRUE(pdf.removeBinning(x));
   EXPECT_FALSE(pdf.isBinnedDistribution(RooArgSet(x)));
   EXPECT_EQ(pdf.getBinning(x), nullptr);

   std::string integratorName;
   const double value = integrate(pdf, x, integratorName);
   EXPECT_EQ(integratorName.find("RooBinIntegrator"), std::string::npos) << integratorName;
   // Same integral, but now via the (approximate) generic numeric integrator.
   EXPECT_NEAR(value, piecewiseFlatIntegral, 1e-2);

   // Removing again (or for an unset/unknown observable) returns false.
   EXPECT_FALSE(pdf.removeBinning(x));
   RooRealVar other("other", "", 0, 1);
   EXPECT_FALSE(pdf.removeBinning(other));
}

TEST(GenericPdf, BinnedIntegrationUniformBinning)
{
   MAKE_PIECEWISE_FLAT_VARS
   RooGenericPdf pdf("pdf", "", piecewiseFlatFormula, RooArgList(x, h0, h1, h2, h3, h4));

   // A RooUniformBinning describes the same bins compactly (just min, max, n).
   pdf.setBinning(x, RooUniformBinning(0.0, 10.0, 5));

   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(x)));

   std::string integratorName;
   const double value = integrate(pdf, x, integratorName);
   EXPECT_NE(integratorName.find("RooBinIntegrator"), std::string::npos) << integratorName;
   EXPECT_DOUBLE_EQ(value, piecewiseFlatIntegral);
}

// The existing tests above all use uniform bins. This one uses bins of
// unequal width to check that a non-uniform binning is integrated correctly:
// the integral must weight each bin's height by its own width.
TEST(GenericPdf, BinnedIntegrationNonUniformBinning)
{
   RooRealVar x("x", "x", 0., 10.);
   RooRealVar b0("b0", "", 1.0), b1("b1", "", 3.0), b2("b2", "", 2.0), b3("b3", "", 4.0);

   // A step function that is flat within the non-uniform bins
   // [0,1), [1,3), [3,6), [6,10] (widths 1, 2, 3, 4).
   const char *formula = "(x<1)*b0 + (x>=1)*(x<3)*b1 + (x>=3)*(x<6)*b2 + (x>=6)*b3";
   RooGenericPdf pdf("pdf", "", formula, RooArgList(x, b0, b1, b2, b3));

   const double boundaries[] = {0, 1, 3, 6, 10};
   pdf.setBinning(x, RooBinning(4, boundaries));

   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(x)));
   const RooAbsBinning *storedBinning = pdf.getBinning(x);
   ASSERT_NE(storedBinning, nullptr);
   ASSERT_EQ(storedBinning->numBoundaries(), 5);
   for (int i = 0; i < storedBinning->numBoundaries(); ++i) {
      EXPECT_DOUBLE_EQ(storedBinning->array()[i], boundaries[i]);
   }

   std::string integratorName;
   const double value = integrate(pdf, x, integratorName);
   EXPECT_NE(integratorName.find("RooBinIntegrator"), std::string::npos) << integratorName;
   // sum(height * width) = 1*1 + 3*2 + 2*3 + 4*4 = 29.
   EXPECT_DOUBLE_EQ(value, 1.0 * 1.0 + 3.0 * 2.0 + 2.0 * 3.0 + 4.0 * 4.0);
}

TEST(GenericPdf, BinnedFlatnessCheck)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::FATAL); // silence the expected error
   MAKE_PIECEWISE_FLAT_VARS
   RooGenericPdf pdf("pdf", "", piecewiseFlatFormula, RooArgList(x, h0, h1, h2, h3, h4));

   // Bins that straddle a jump ([1,3], [3,5], ... cross the steps at 2, 4, ...)
   // are not flat: the binning must be rejected.
   const double bad[] = {1, 3, 5, 7, 9};
   pdf.setBinning(x, RooBinning(4, bad));
   EXPECT_FALSE(pdf.isBinnedDistribution(RooArgSet(x)));

   // With the flatness check disabled, the binning is accepted regardless.
   pdf.setBinning(x, RooBinning(4, bad), /*checkFlatness=*/false);
   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(x)));
}

TEST(GenericPdf, BinnedNonFormulaVariable)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::FATAL); // silence the expected error
   MAKE_PIECEWISE_FLAT_VARS
   RooGenericPdf pdf("pdf", "", piecewiseFlatFormula, RooArgList(x, h0, h1, h2, h3, h4));

   // An observable that is not one of the formula variables must be rejected.
   RooRealVar other("other", "", 0, 1);
   pdf.setBinning(other, RooUniformBinning(0.0, 1.0, 5));
   EXPECT_FALSE(pdf.isBinnedDistribution(RooArgSet(other)));
}

// isBinnedDistribution() must ignore observables that the pdf does not depend
// on: the function is constant, hence trivially binned, in them. Otherwise an
// unrelated observable in the query set would spuriously report the pdf as
// unbinned.
TEST(GenericPdf, BinnedIgnoresUnrelatedObservable)
{
   MAKE_PIECEWISE_FLAT_VARS
   RooGenericPdf pdf("pdf", "", piecewiseFlatFormula, RooArgList(x, h0, h1, h2, h3, h4));
   pdf.setBinning(x, RooUniformBinning(0.0, 10.0, 5));

   RooRealVar y("y", "y", 0., 4.); // not a formula variable of the pdf
   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(x, y)));
}

// A RooGenericPdf that is flat-binned in its own observable must work as a
// binned component of a RooProduct. RooProduct queries each component with the
// full observable set, so each component must ignore the observable it does not
// depend on (see BinnedIgnoresUnrelatedObservable) for the product to report as
// binned.
TEST(GenericPdf, BinnedProductComposition)
{
   RooRealVar x("x", "x", 0., 10.);
   RooRealVar y("y", "y", 0., 4.);
   RooRealVar a("a", "", 1.0), b("b", "", 2.0);

   RooGenericPdf px("px", "", "a*(x<100)", RooArgList(x, a));
   RooGenericPdf py("py", "", "b*(y<100)", RooArgList(y, b));
   px.setBinning(x, RooUniformBinning(0.0, 10.0, 5));
   py.setBinning(y, RooUniformBinning(0.0, 4.0, 2));

   EXPECT_TRUE(px.isBinnedDistribution(RooArgSet(x, y)));
   EXPECT_TRUE(py.isBinnedDistribution(RooArgSet(x, y)));

   RooProduct prod("prod", "", RooArgList(px, py));
   EXPECT_TRUE(prod.isBinnedDistribution(RooArgSet(x, y)));
}

// The lookup resolves the observable by name, so a different object with the
// same name (e.g. one read back separately from a file) resolves to the same
// binning - it need not be the pdf's own internal server object.
TEST(GenericPdf, BinnedLookupBySameNamedVariable)
{
   MAKE_PIECEWISE_FLAT_VARS
   RooGenericPdf pdf("pdf", "", piecewiseFlatFormula, RooArgList(x, h0, h1, h2, h3, h4));
   pdf.setBinning(x, RooUniformBinning(0.0, 10.0, 5));

   RooRealVar xStandIn("x", "x", 0., 10.);
   EXPECT_NE(&xStandIn, pdf.getParameter("x"));

   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(xStandIn)));
   EXPECT_EQ(pdf.getBinning(xStandIn), pdf.getBinning(x));

   std::unique_ptr<std::list<double>> boundaries{pdf.binBoundaries(xStandIn, 0.0, 10.0)};
   ASSERT_NE(boundaries, nullptr);
   EXPECT_EQ(boundaries->size(), 6u); // 5 bins -> 6 boundaries
}

TEST(GenericPdf, FormulaVarBinningAccessor)
{
   RooRealVar x("x", "x", 0., 10.);
   RooFormulaVar formula("formula", "floor(x)", RooArgList(x));
   const double boundaries[] = {-1.0, 1.0, 4.0, 11.0};
   formula.setBinning(x, RooBinning(3, boundaries), /*checkFlatness=*/false);

   RooRealVar xStandIn("x", "x", 0., 10.);
   const RooAbsBinning *storedBinning = formula.getBinning(xStandIn);
   ASSERT_NE(storedBinning, nullptr);
   ASSERT_EQ(storedBinning->numBoundaries(), 4);
   for (int i = 0; i < storedBinning->numBoundaries(); ++i) {
      EXPECT_DOUBLE_EQ(storedBinning->array()[i], boundaries[i]);
   }

   EXPECT_TRUE(formula.removeBinning(xStandIn));
   EXPECT_EQ(formula.getBinning(x), nullptr);
}

TEST(GenericPdf, BinnedMultipleObservables)
{
   MAKE_PIECEWISE_FLAT_VARS
   // Second observable, flat in two bins on [0, 4].
   RooRealVar y("y", "y", 0., 4.);
   RooRealVar g0("g0", "", 2.0), g1("g1", "", 5.0);

   TString formula = TString::Format("(%s) * ((floor(y/2)==0)*g0 + (floor(y/2)==1)*g1)", piecewiseFlatFormula);
   RooGenericPdf pdf("pdf", "", formula, RooArgList(x, y, h0, h1, h2, h3, h4, g0, g1));

   pdf.setBinning(x, RooUniformBinning(0.0, 10.0, 5));

   // Only x is binned so far: the joint {x,y} distribution is not fully binned.
   EXPECT_FALSE(pdf.isBinnedDistribution(RooArgSet(x, y)));
   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(x)));

   pdf.setBinning(y, RooUniformBinning(0.0, 4.0, 2));
   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(x, y)));

   std::string integratorName;
   const double value = integrate(pdf, RooArgSet(x, y), integratorName);
   EXPECT_NE(integratorName.find("RooBinIntegrator"), std::string::npos) << integratorName;
   // int_x g(x) dx = 23, int_y g(y) dy = (2+5)*2 = 14, product = 322.
   EXPECT_DOUBLE_EQ(value, piecewiseFlatIntegral * ((2.0 + 5.0) * 2.0));
}

// The binning is keyed by the observable's index in the internal variable list,
// not by its name, so renaming the observable after registering the binning
// must not lose it.
TEST(GenericPdf, BinnedBoundariesSurviveRename)
{
   MAKE_PIECEWISE_FLAT_VARS
   RooGenericPdf pdf("pdf", "", piecewiseFlatFormula, RooArgList(x, h0, h1, h2, h3, h4));
   pdf.setBinning(x, RooUniformBinning(0.0, 10.0, 5));
   ASSERT_TRUE(pdf.isBinnedDistribution(RooArgSet(x)));

   // Rename the observable in place.
   x.SetName("xRenamed");

   EXPECT_TRUE(pdf.isBinnedDistribution(RooArgSet(x)));

   std::string integratorName;
   const double value = integrate(pdf, RooArgSet(x), integratorName);
   EXPECT_NE(integratorName.find("RooBinIntegrator"), std::string::npos) << integratorName;
   EXPECT_DOUBLE_EQ(value, piecewiseFlatIntegral);
}

// Ensure the implementation of RooGenericPdf::binBoundaries() (and
// RooFormulaVar::binBoundaries()) is consistent with equivalent RooHistPdf.
TEST(GenericPdf, BinnedBoundariesConsistentWithHistPdf)
{
   int nBins = 5;

   RooRealVar x{"x", "x", 0, 0, 5};
   x.setBins(nBins);

   RooDataHist dh{"dh", "", x};

   RooHistPdf hpdf{"hpdf", "", x, dh};
   RooGenericPdf gpdf{"gpdf", "x[0] - x[0] + 1", {x}}; // uniform dummy
   gpdf.setBinning(x, x.getBinning());

   RooFormulaVar fvar{"fvar", "x[0] - x[0] + 1", {x}}; // uniform dummy
   fvar.setBinning(x, x.getBinning());

   // intentionally beyond the bin boundaries
   double lo = -10;
   double hi = 10;

   std::unique_ptr<std::list<double>> boundsHistPdf{hpdf.binBoundaries(x, lo, hi)};
   std::unique_ptr<std::list<double>> boundsGenericPdf{gpdf.binBoundaries(x, lo, hi)};
   std::unique_ptr<std::list<double>> boundsFormulaVar{fvar.binBoundaries(x, lo, hi)};

   EXPECT_EQ(boundsHistPdf->size(), nBins + 1);
   EXPECT_EQ(boundsGenericPdf->size(), nBins + 1);
   EXPECT_EQ(boundsFormulaVar->size(), nBins + 1);

   auto iterHistPdf = boundsHistPdf->begin();
   auto iterGenericPdf = boundsGenericPdf->begin();
   auto iterFormulaVar = boundsFormulaVar->begin();

   for (int i = 0; i < nBins + 1; ++i) {
      EXPECT_EQ(*iterHistPdf, static_cast<double>(i));
      EXPECT_EQ(*iterGenericPdf, static_cast<double>(i));
      EXPECT_EQ(*iterFormulaVar, static_cast<double>(i));
      iterHistPdf++;
      iterGenericPdf++;
      iterFormulaVar++;
   }
}

// Regression test for https://github.com/root-project/root/issues/21371:
// an unused parameter (b) is pruned, so the persisted @N indices must be
// remapped or the formula silently mismaps after a write/read cycle.
TEST(GenericPdf, SerializationWithUnusedParam)
{
   RooWorkspace w("w");
   w.factory("a[2,-10,10]");
   w.factory("b[99,-10,10]");
   w.factory("c[3,-10,10]");
   w.factory("d[4,-10,10]");
   w.factory("EXPR::pdf('@0*@2+d', a, b, c, d)");

   TString fn = "RooGenericPdfSerialization.root";
   w.writeToFile(fn);
   TFile fin(fn);
   RooWorkspace *w2 = nullptr;
   fin.GetObject("w", w2);
   ASSERT_NE(w2, nullptr);
   auto *pdf = static_cast<RooAbsReal *>(w2->pdf("pdf"));

   // If @2 still maps to c, changing c updates pdf = a*c + d = 2*5 + 4 = 14.
   static_cast<RooRealVar *>(w2->var("c"))->setVal(5.0);
   EXPECT_DOUBLE_EQ(pdf->getVal(), 2.0 * 5.0 + 4.0);
}
