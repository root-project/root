// Tests for the RooNaNPacker
// Authors: Stephan Hageboeck, CERN  04/2020
#include "RooNaNPacker.h"
#include "RooRealVar.h"
#include "RooGenericPdf.h"
#include "RooMinimizer.h"
#include "RooFitResult.h"
#include "RooDataSet.h"

#include "gtest/gtest.h"

#include <cmath>
#include <bitset>
#include <cstdint>

void dumpFloats(double val) {
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
  if (dump) dumpFloats(rnp._payload);

  // Create a normal double
  rnp._payload = 1.337;
  EXPECT_FALSE(std::isnan(rnp._payload));
  EXPECT_FALSE(rnp.isNaNWithPayload());
  EXPECT_DOUBLE_EQ(rnp._payload, 1.337);
  EXPECT_FLOAT_EQ(rnp.getPayload(), 0.f);
  if (dump) dumpFloats(rnp._payload);

  // Create a normal double
  rnp._payload = 4.;
  EXPECT_FALSE(std::isnan(rnp._payload));
  EXPECT_FALSE(rnp.isNaNWithPayload());
  EXPECT_DOUBLE_EQ(rnp._payload, 4.);
  EXPECT_FLOAT_EQ(rnp.getPayload(), 0.f);
  if (dump) dumpFloats(rnp._payload);

  // Create a simple payload
  rnp.setPayload(0.);
  EXPECT_TRUE(std::isnan(rnp._payload));
  EXPECT_TRUE(rnp.isNaNWithPayload());
  EXPECT_FLOAT_EQ(rnp.getPayload(), 0.f);
  if (dump) dumpFloats(rnp._payload);

  // Create a simple payload
  rnp.setPayload(2.);
  EXPECT_TRUE(std::isnan(rnp._payload));
  EXPECT_TRUE(rnp.isNaNWithPayload());
  EXPECT_FLOAT_EQ(rnp.getPayload(), 2.f);
  if (dump) dumpFloats(rnp._payload);

  // Create a simple payload
  rnp.setPayload(4.);
  EXPECT_TRUE(std::isnan(rnp._payload));
  EXPECT_TRUE(rnp.isNaNWithPayload());
  EXPECT_FLOAT_EQ(rnp.getPayload(), 4.f);
  if (dump) dumpFloats(rnp._payload);

  // Create a NaN that doesn't have the magic tag,
  // so no information encoded
  rnp._payload = std::numeric_limits<double>::quiet_NaN();
  const float tmp = 1234.5f;
  std::memcpy(&rnp._payload, &tmp, sizeof(float));
  EXPECT_TRUE(std::isnan(rnp._payload));
  EXPECT_FALSE(rnp.isNaNWithPayload());
  EXPECT_FLOAT_EQ(rnp.getPayload(), 0.f);
  if (dump) dumpFloats(rnp._payload);

}


TEST(RooNaNPacker, FitSimpleLinear) {
  RooRealVar x("x", "x", -10, 10);
  RooRealVar a1("a1", "a1", 12., -5., 15.);
  RooGenericPdf pdf("pdf", "a1 + x", RooArgSet(x, a1));
  std::unique_ptr<RooDataSet> data(pdf.generate(x, 1000));
  std::unique_ptr<RooAbsReal> nll(pdf.createNLL(*data));

  ASSERT_FALSE(std::isnan(pdf.getVal(RooArgSet(x))));
  a1.setVal(-9.);
  ASSERT_TRUE(std::isnan(pdf.getVal(RooArgSet(x))));

  RooMinimizer minim(*nll);
  minim.migrad();
  minim.hesse();
  auto fitResult = minim.save();

  EXPECT_EQ(fitResult->status(), 0);
  EXPECT_NEAR(a1.getVal(), 12., a1.getError());
}



TEST(RooNaNPacker, FitParabola) {
  RooRealVar x("x", "x", -10, 10);
  RooRealVar a1("a1", "a1", 12., -10., 15.);
  RooRealVar a2("a2", "a2", 1.1, -5., 15.);
  RooGenericPdf pdf("pdf", "a1 + x + a2 *x*x", RooArgSet(x, a1, a2));
  std::unique_ptr<RooDataSet> data(pdf.generate(x, 10000));
  auto nll = pdf.createNLL(*data);

  RooArgSet params(a1, a2);
  RooArgSet evilValues;
  a1.setVal(-9.);
  a2.setVal(-1.);
  params.snapshot(evilValues);

  params = evilValues;
  auto fitResult1 = pdf.fitTo(*data, RooFit::Save());

  params = evilValues;

  auto fitResult2 = pdf.fitTo(*data, RooFit::Save());

  fitResult1->Print();
  fitResult2->Print();

  for (auto fitResult : std::initializer_list<RooFitResult*>{fitResult1, fitResult2}) {
    std::string config = (fitResult == fitResult1 ? "No error wall" : "Error wall");
    const auto& a1Final = static_cast<RooRealVar&>(fitResult->floatParsFinal()[0]);
    const auto& a2Final = static_cast<RooRealVar&>(fitResult->floatParsFinal()[1]);
    EXPECT_EQ(fitResult->status(), 0) << config;
    EXPECT_NEAR(a1Final.getVal(), 12., a1Final.getError()) << config;
    EXPECT_NEAR(a2Final.getVal(),  0., a2Final.getError()) << config;
  }

  EXPECT_LT(fitResult1->numInvalidNLL(), fitResult2->numInvalidNLL());
}

