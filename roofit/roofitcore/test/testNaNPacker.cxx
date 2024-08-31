// Tests for the RooNaNPacker
// Authors: Stephan Hageboeck, CERN  04/2020

#include <RooAddPdf.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooNaNPacker.h>
#include <RooRandom.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>

#include "gtest_wrapper.h"

#include <cmath>
#include <cstdint>
#include <bitset>

void dumpFloats(double val)
{
   float tmp[2];
   std::memcpy(tmp, &val, sizeof(double));

   for (int i = 1; i >= 0; --i) {
      unsigned long long ull = 0ull;
      std::memcpy(&ull, &tmp[i], 4);
      std::bitset<32> bits(ull);
      std::cout << bits << " " << std::flush;
   }
   std::cout << std::endl;
}

// Test if we can pack floats into NaNs, and recover them.
TEST(RooNaNPacker, CanPackStuffIntoNaNs)
{
   static_assert((RooNaNPacker::magicTag & RooNaNPacker::magicTagMask) == RooNaNPacker::magicTag, "Bit mask wrong.");
   constexpr bool dump = false;

   // Create a NaN that has 1.337f as payload
   RooNaNPacker rnp;
   rnp.setPayload(1.337f);
   EXPECT_TRUE(std::isnan(rnp._payload));
   EXPECT_TRUE(rnp.isNaNWithPayload());
   EXPECT_FLOAT_EQ(rnp.getPayload(), 1.337f);
   if (dump)
      dumpFloats(rnp._payload);

   // Create a normal double
   rnp._payload = 1.337;
   EXPECT_FALSE(std::isnan(rnp._payload));
   EXPECT_FALSE(rnp.isNaNWithPayload());
   EXPECT_DOUBLE_EQ(rnp._payload, 1.337);
   EXPECT_FLOAT_EQ(rnp.getPayload(), 0.f);
   if (dump)
      dumpFloats(rnp._payload);

   // Create a normal double
   rnp._payload = 4.;
   EXPECT_FALSE(std::isnan(rnp._payload));
   EXPECT_FALSE(rnp.isNaNWithPayload());
   EXPECT_DOUBLE_EQ(rnp._payload, 4.);
   EXPECT_FLOAT_EQ(rnp.getPayload(), 0.f);
   if (dump)
      dumpFloats(rnp._payload);

   // Create a simple payload
   rnp.setPayload(0.);
   EXPECT_TRUE(std::isnan(rnp._payload));
   EXPECT_TRUE(rnp.isNaNWithPayload());
   EXPECT_FLOAT_EQ(rnp.getPayload(), 0.f);
   if (dump)
      dumpFloats(rnp._payload);

   // Create a simple payload
   rnp.setPayload(2.);
   EXPECT_TRUE(std::isnan(rnp._payload));
   EXPECT_TRUE(rnp.isNaNWithPayload());
   EXPECT_FLOAT_EQ(rnp.getPayload(), 2.f);
   if (dump)
      dumpFloats(rnp._payload);

   // Create a simple payload
   rnp.setPayload(4.);
   EXPECT_TRUE(std::isnan(rnp._payload));
   EXPECT_TRUE(rnp.isNaNWithPayload());
   EXPECT_FLOAT_EQ(rnp.getPayload(), 4.f);
   if (dump)
      dumpFloats(rnp._payload);

   // Create a NaN that doesn't have the magic tag,
   // so no information encoded
   rnp._payload = std::numeric_limits<double>::quiet_NaN();
   const float tmp = 1234.5f;
   std::memcpy(&rnp._payload, &tmp, sizeof(float));
   EXPECT_TRUE(std::isnan(rnp._payload));
   EXPECT_FALSE(rnp.isNaNWithPayload());
   EXPECT_FLOAT_EQ(rnp.getPayload(), 0.f);
   if (dump)
      dumpFloats(rnp._payload);
}

#if !defined(_MSC_VER) || defined(_WIN64) || defined(NDEBUG) || defined(R__ENABLE_BROKEN_WIN_TESTS)

// Demonstrate value preserving behavior after arithmetic on packed NaNs.
TEST(RooNaNPacker, PackedNaNPreservedAfterArithmetic)
{
   RooNaNPacker rnp, rnp2;
   rnp.setPayload(1.337f);
   EXPECT_TRUE(rnp.isNaNWithPayload());

   // multiply the packed NaN by 1 and use the result as rnp2's NaN with payload
   rnp2._payload = 1. * rnp.getNaNWithPayload();
   EXPECT_TRUE(rnp2.isNaNWithPayload());
   EXPECT_EQ(rnp.getPayload(), rnp2.getPayload());

   // multiply the packed NaN by -1
   rnp2._payload = -1. * rnp.getNaNWithPayload();
   EXPECT_TRUE(rnp2.isNaNWithPayload());
   // minus signs on the NaN don't affect the payload
   EXPECT_EQ(rnp.getPayload(), rnp2.getPayload());

   // multiply the packed NaN by 4242
   rnp2._payload = 4242. * rnp.getNaNWithPayload();
   EXPECT_TRUE(rnp2.isNaNWithPayload());
   // random multiplicative values on the NaN don't affect it all either
   EXPECT_EQ(rnp.getPayload(), rnp2.getPayload());

   // add 4242 to the packed NaN
   rnp2._payload = rnp.getNaNWithPayload() + 4242.;
   EXPECT_TRUE(rnp2.isNaNWithPayload());
   // addition also has no effect, the NaN retains its bits
   EXPECT_EQ(rnp.getPayload(), rnp2.getPayload());

   // divide packed NaN by 1337, subtract 20 and take the modulo of 38 before calculating the sine of all this
   rnp2._payload = std::sin(std::fmod((rnp.getNaNWithPayload() / 1337. - 20.), 38.));
   EXPECT_TRUE(rnp2.isNaNWithPayload());
   // nothing can harm the PackedNaN
   EXPECT_EQ(rnp.getPayload(), rnp2.getPayload());
}

#endif // !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)

/// Fit a simple linear function, that starts in the negative.
TEST(RooNaNPacker, FitSimpleLinear)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x("x", "x", -10, 10);
   RooRealVar a1("a1", "a1", 12., -5., 15.);
   RooGenericPdf pdf("pdf", "a1 + x", {x, a1});
   std::unique_ptr<RooDataSet> data(pdf.generate(x, 1000));
   std::unique_ptr<RooAbsReal> nll(pdf.createNLL(*data));

   RooArgSet normSet{x};
   ASSERT_FALSE(std::isnan(pdf.getVal(normSet)));
   a1.setVal(-9.);
   ASSERT_TRUE(std::isnan(pdf.getVal(normSet)));

   RooMinimizer minim(*nll);
   minim.setPrintLevel(-1);
   minim.setPrintEvalErrors(-1);
   minim.migrad();
   minim.hesse();
   std::unique_ptr<RooFitResult> fitResult{minim.save()};

   EXPECT_EQ(fitResult->status(), 0);
   EXPECT_NEAR(a1.getVal(), 12., a1.getError());
}

class TestForDifferentBackends : public testing::TestWithParam<std::tuple<RooFit::EvalBackend>> {
public:
   TestForDifferentBackends() : _evalBackend{RooFit::EvalBackend::Legacy()} {}

private:
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(1337ul);
      _evalBackend = std::get<0>(GetParam());
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   RooFit::EvalBackend _evalBackend;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

/// Fit a parabola, where parameters are set up such that negative function values are obtained.
/// The minimiser needs to recover from that.
/// Test also that when recovery with NaN packing is switched off, the minimiser fails to recover.
TEST_P(TestForDifferentBackends, FitParabola)
{
   RooRealVar x("x", "x", -10, 10);
   RooRealVar a1("a1", "a1", 12., -10., 20.);
   RooRealVar a2("a2", "a2", 1.1, -10., 20.);
   RooGenericPdf pdf("pdf", "a1 + x + a2 *x*x", {x, a1, a2});
   std::unique_ptr<RooDataSet> data(pdf.generate(x, 10000));

   RooArgSet params(a1, a2);
   RooArgSet paramsInit;
   params.snapshot(paramsInit);
   RooArgSet evilValues;
   a1.setVal(-9.);
   a2.setVal(-1.);
   params.snapshot(evilValues);

   using namespace RooFit;
   params.assign(evilValues);
   // Note: we can't use Hesse or Minos. Without the error recovery, Minuit2
   // would print errors that would cause the unit test to fail.
   std::unique_ptr<RooFitResult> fitResultOld(
      pdf.fitTo(*data, RecoverFromUndefinedRegions(0.), Save(),
                PrintLevel(-1),      // Don't need fit status printed
                PrintEvalErrors(-1), // We provoke a lot of evaluation errors in this test. Don't print those.
                _evalBackend, Hesse(false), Minos(false)));

   params.assign(evilValues);
   std::unique_ptr<RooFitResult> fitResultNew(
      pdf.fitTo(*data, Save(), PrintLevel(-1), PrintEvalErrors(-1), _evalBackend));

   ASSERT_NE(fitResultOld, nullptr);
   ASSERT_NE(fitResultNew, nullptr);

   const auto &a1Old = static_cast<RooRealVar &>(fitResultOld->floatParsFinal()[0]);
   const auto &a2Old = static_cast<RooRealVar &>(fitResultOld->floatParsFinal()[1]);

   const auto &a1Recover = static_cast<RooRealVar &>(fitResultNew->floatParsFinal()[0]);
   const auto &a2Recover = static_cast<RooRealVar &>(fitResultNew->floatParsFinal()[1]);

   EXPECT_NE(fitResultOld->status(), 0);
   EXPECT_EQ(fitResultNew->status(), 0);
   EXPECT_NEAR(a1Recover.getVal(), static_cast<RooAbsReal &>(paramsInit["a1"]).getVal(), a1Recover.getError());
   EXPECT_NEAR(a2Recover.getVal(), static_cast<RooAbsReal &>(paramsInit["a2"]).getVal(), a2Recover.getError());

   EXPECT_LT(a1Old.getVal(), 0.);
   EXPECT_LT(a2Old.getVal(), 0.);

   // In the past, when Minuit2 was not the default minimizer yet, there was
   // also a check that the number of invalid NLL evaluations was reduced with
   // the error recovery:
   //
   //   EXPECT_LT(fitResultNew->numInvalidNLL(), fitResultOld->numInvalidNLL());
   //
   // However, Minuit2 takes less evaluations to realize that the minimization
   // without error recovery is hopeless, resulting in less invalid NLL
   // evaluations when the error recovery is off. Hence, the comparison is not
   // meaningful and was commended out.
}

INSTANTIATE_TEST_SUITE_P(RooNaNPacker, TestForDifferentBackends, testing::Values(ROOFIT_EVAL_BACKENDS),
                         [](testing::TestParamInfo<TestForDifferentBackends::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            return ss.str();
                         });

#undef BATCH_MODE_VALS

/// Make coefficients of RooAddPdf sum to more than 1. Fitter should recover from this.
TEST(RooNaNPacker, FitAddPdf_DegenerateCoeff)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   constexpr bool verbose = false;
   RooRandom::randomGenerator()->SetSeed(100);

   RooRealVar x("x", "x", 0., 10);
   RooRealVar a1("a1", "a1", 0.4, -10., 10.);
   RooRealVar a2("a2", "a2", 0.4, -10., 10.);
   RooGenericPdf pdf1("gen1", "exp(-2.*x)", {x});
   RooGenericPdf pdf2("gen2", "TMath::Gaus(x, 3, 2)", {x});
   RooGenericPdf pdf3("gen3", "x*x*x+1", {x});
   RooAddPdf pdf("sum", "a1*gen1 + a2*gen2 + (1-a1-a2)*gen3", {pdf1, pdf2, pdf3}, {a1, a2});
   std::unique_ptr<RooDataSet> data(pdf.generate(x, 2000));
   std::unique_ptr<RooAbsReal> nll{pdf.createNLL(*data)};

   RooArgSet params(a1, a2);
   RooArgSet paramsInit;
   params.snapshot(paramsInit);

   RooArgSet evilValues;
   a1.setVal(0.6);
   a2.setVal(0.7);
   params.snapshot(evilValues);

   params.assign(evilValues);

   std::unique_ptr<RooFitResult> fitResult1;
   std::unique_ptr<RooFitResult> fitResult2;
   for (auto tryRecover : std::initializer_list<double>{0., 10.}) {
      params.assign(evilValues);

      RooMinimizer minim(*nll);
      minim.setRecoverFromNaNStrength(tryRecover);
      minim.setPrintLevel(-1);
      minim.setPrintEvalErrors(-1);
      minim.migrad();
      std::unique_ptr<RooFitResult> fitResult{minim.save()};

      const auto &a1Final = static_cast<RooRealVar &>(fitResult->floatParsFinal()[0]);
      const auto &a2Final = static_cast<RooRealVar &>(fitResult->floatParsFinal()[1]);

      // Only the fit with error recovery should have status zero.
      EXPECT_EQ(fitResult->status() == 0, tryRecover != 0.0);

      if (tryRecover != 0.) {
         EXPECT_EQ(fitResult->status(), 0) << "Recovery strength=" << tryRecover;
         EXPECT_NEAR(a1Final.getVal(), static_cast<RooAbsReal &>(paramsInit["a1"]).getVal(), a1Final.getError())
            << "Recovery strength=" << tryRecover;
         EXPECT_NEAR(a2Final.getVal(), static_cast<RooAbsReal &>(paramsInit["a2"]).getVal(), a2Final.getError())
            << "Recovery strength=" << tryRecover;
         EXPECT_NEAR(a1Final.getVal() + a2Final.getVal(), 0.8, 0.02) << "Check that coefficients sum to 1. "
                                                                     << "Recovery strength=" << tryRecover;
      }

      if (verbose) {
         std::cout << "Recovery strength:" << tryRecover << "\n";
         fitResult->Print();
      }

      (tryRecover != 0. ? fitResult1 : fitResult2) = std::move(fitResult);
   }

   // This makes clang-tidy happy:
   ASSERT_NE(fitResult1, nullptr);
   ASSERT_NE(fitResult2, nullptr);
}

/// Make coefficients of RooRealSumPdf sum to more than 1. Fitter should recover from this.
TEST(RooNaNPacker, Interface_RooAbsPdf_fitTo_RooRealSumPdf_DegenerateCoeff)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   constexpr bool verbose = false;
   RooRandom::randomGenerator()->SetSeed(100);

   RooRealVar x("x", "x", 0., 10);
   RooRealVar a1("a1", "a1", 0.3, -10., 10.);
   RooRealVar a2("a2", "a2", 0.4, -10., 10.);
   RooGenericPdf pdf1("gen1", "exp(-0.5*x)", {x});
   RooGenericPdf pdf2("gen2", "TMath::Gaus(x, 5, 0.7)", {x});
   RooGenericPdf pdf3("gen3", "TMath::Gaus(x, 8, 0.8)", {x});
   RooRealSumPdf pdf("sum", "a1*gen1 + a2*gen2 + (1-a1-a2)*gen3", {pdf1, pdf2, pdf3}, {a1, a2});
   std::unique_ptr<RooDataSet> data(pdf.generate(x, 5000));

   RooArgSet params(a1, a2);
   RooArgSet paramsInit;
   params.snapshot(paramsInit);

   RooArgSet evilValues;
   a1.setVal(0.6);
   a2.setVal(0.7);
   params.snapshot(evilValues);

   params.assign(evilValues);

   std::unique_ptr<RooFitResult> fitResult1;
   std::unique_ptr<RooFitResult> fitResult2;
   for (auto tryRecover : std::initializer_list<double>{0., 10.}) {
      params.assign(evilValues);

      using namespace RooFit;
      // Note: we can't do Hesse here. Without the error recovery, Minuit2
      // would print errors that would cause the unit test to fail.
      std::unique_ptr<RooFitResult> fitResult{pdf.fitTo(*data, PrintLevel(-1), PrintEvalErrors(-1), Save(),
                                                        RecoverFromUndefinedRegions(tryRecover), Hesse(false))};

      const auto &a1Final = static_cast<RooRealVar &>(fitResult->floatParsFinal()[0]);
      const auto &a2Final = static_cast<RooRealVar &>(fitResult->floatParsFinal()[1]);

      // Only the fit with error recovery should have status zero.
      EXPECT_EQ(fitResult->status() == 0, tryRecover != 0.0);

      if (tryRecover != 0.) {
         EXPECT_EQ(fitResult->status(), 0) << "Recovery strength=" << tryRecover;
         EXPECT_NEAR(a1Final.getVal(), static_cast<RooAbsReal &>(paramsInit["a1"]).getVal(), 3. * a1Final.getError())
            << "Recovery strength=" << tryRecover;
         EXPECT_NEAR(a2Final.getVal(), static_cast<RooAbsReal &>(paramsInit["a2"]).getVal(), 3. * a2Final.getError())
            << "Recovery strength=" << tryRecover;
         EXPECT_GE(a1Final.getVal() + a2Final.getVal(), 0.) << "Check that coefficients are in [0, 1]. "
                                                            << "Recovery strength=" << tryRecover;
         EXPECT_LE(a1Final.getVal() + a2Final.getVal(), 1.) << "Check that coefficients sum to [0, 1]. "
                                                            << "Recovery strength=" << tryRecover;
      }

      if (verbose) {
         std::cout << "Recovery strength:" << tryRecover << "\n";
         fitResult->Print();
      }

      (tryRecover != 0. ? fitResult1 : fitResult2) = std::move(fitResult);
   }

   // This makes clang-tidy happy:
   ASSERT_NE(fitResult1, nullptr);
   ASSERT_NE(fitResult2, nullptr);
}
