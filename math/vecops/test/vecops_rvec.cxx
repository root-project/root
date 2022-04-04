#include <gtest/gtest.h>
#include <Math/LorentzVector.h>
#include <Math/PtEtaPhiM4D.h>
#include <Math/Vector4Dfwd.h>
#include <ROOT/RVec.hxx>
#include <ROOT/TSeq.hxx>
#include <TFile.h>
#include <TInterpreter.h>
#include <TTree.h>
#include <TSystem.h>
#include <TLorentzVector.h>
#include <vector>
#include <sstream>
#include <cmath>

using namespace ROOT;
using namespace ROOT::VecOps;

void CheckEqual(const ROOT::RVecF &a, const ROOT::RVecF &b, std::string_view msg = "")
{
   const auto asize = a.size();
   const auto bsize = b.size();
   EXPECT_EQ(asize, bsize);
   for (unsigned int i = 0; i < asize; ++i) {
      EXPECT_FLOAT_EQ(a[i], b[i]) << msg;
   }
}

void CheckEqual(const ROOT::RVecD &a, const ROOT::RVecD &b, std::string_view msg = "")
{
   const auto asize = a.size();
   const auto bsize = b.size();
   EXPECT_EQ(asize, bsize);
   for (unsigned int i = 0; i < asize; ++i) {
      EXPECT_DOUBLE_EQ(a[i], b[i]) << msg;
   }
}

void CheckEqual(const ROOT::Math::PtEtaPhiMVector &a, const ROOT::Math::PtEtaPhiMVector &b) {
   EXPECT_DOUBLE_EQ(a.Pt(), b.Pt());
   EXPECT_DOUBLE_EQ(a.Eta(), b.Eta());
   EXPECT_DOUBLE_EQ(a.Phi(), b.Phi());
   EXPECT_DOUBLE_EQ(a.M(), b.M());
}

template <typename T, typename V>
void CheckEqual(const T &a, const V &b, std::string_view msg = "")
{
   const auto asize = a.size();
   const auto bsize = b.size();
   EXPECT_EQ(asize, bsize);
   for (unsigned int i = 0; i < asize; ++i) {
      EXPECT_EQ(a[i], b[i]) << msg;
   }
}

TEST(VecOps, DefaultCtor)
{
   RVec<int> v;
   EXPECT_EQ(v.size(), 0u);
}

TEST(VecOps, InitListCtor)
{
   RVec<int> v{1, 2, 3};
   EXPECT_EQ(v.size(), 3u);
   EXPECT_EQ(v[0], 1);
   EXPECT_EQ(v[1], 2);
   EXPECT_EQ(v[2], 3);
}

TEST(VecOps, CopyCtor)
{
   RVec<int> v1{1, 2, 3};
   RVec<int> v2(v1);
   EXPECT_EQ(v1.size(), 3u);
   EXPECT_EQ(v2.size(), 3u);
   EXPECT_EQ(v2[0], 1);
   EXPECT_EQ(v2[1], 2);
   EXPECT_EQ(v2[2], 3);
}

class TLeakChecker {
public:
   static bool fgDestroyed;
   ~TLeakChecker(){
      fgDestroyed = true;
   }
};
bool TLeakChecker::fgDestroyed = false;

// This test is here to check that emplace_back works
// also for T==bool. Indeed, on some platform, vector<bool>
// has no emplace_back. Notable examples are osx 10.14 and gcc 4.8
TEST(VecOps, EmplaceBack)
{
   ROOT::RVec<bool> vb; vb.emplace_back(true);
   ROOT::RVec<int> vi; vi.emplace_back(1);
}

TEST(VecOps, CopyCtorCheckNoLeak)
{
   RVec<TLeakChecker> ref;
   ref.emplace_back(TLeakChecker());
   RVec<TLeakChecker> proxy(ref.data(), ref.size());
   TLeakChecker::fgDestroyed = false;
   {
      auto v = proxy;
   }
   EXPECT_TRUE(TLeakChecker::fgDestroyed);

   TLeakChecker::fgDestroyed = false;
   ref.clear();
   EXPECT_TRUE(TLeakChecker::fgDestroyed);

}

TEST(VecOps, MoveCtor)
{
   RVec<int> v1{1, 2, 3};
   RVec<int> v2(std::move(v1));
   EXPECT_EQ(v2.size(), 3u);
}

TEST(VecOps, Conversion)
{
   RVec<float> fvec{1.0f, 2.0f, 3.0f};
   RVec<unsigned> uvec{1u, 2u, 3u};

   RVec<int>  ivec = uvec;
   RVec<long> lvec = ivec;

   EXPECT_EQ(1, ivec[0]);
   EXPECT_EQ(2, ivec[1]);
   EXPECT_EQ(3, ivec[2]);
   EXPECT_EQ(3u, ivec.size());
   EXPECT_EQ(1l, lvec[0]);
   EXPECT_EQ(2l, lvec[1]);
   EXPECT_EQ(3l, lvec[2]);
   EXPECT_EQ(3u, lvec.size());

   auto dvec1 = RVec<double>(fvec);
   auto dvec2 = RVec<double>(uvec);

   EXPECT_EQ(1.0, dvec1[0]);
   EXPECT_EQ(2.0, dvec1[1]);
   EXPECT_EQ(3.0, dvec1[2]);
   EXPECT_EQ(3u, dvec1.size());
   EXPECT_EQ(1.0, dvec2[0]);
   EXPECT_EQ(2.0, dvec2[1]);
   EXPECT_EQ(3.0, dvec2[2]);
   EXPECT_EQ(3u, dvec2.size());
}

TEST(VecOps, ArithmeticsUnary)
{
   RVec<int> ivec{1, 2, 3};
   RVec<int> pvec = +ivec;
   RVec<int> nvec = -ivec;
   RVec<int> tvec = ~ivec;

   EXPECT_EQ(1, pvec[0]);
   EXPECT_EQ(2, pvec[1]);
   EXPECT_EQ(3, pvec[2]);
   EXPECT_EQ(3u, pvec.size());

   EXPECT_EQ(-1, nvec[0]);
   EXPECT_EQ(-2, nvec[1]);
   EXPECT_EQ(-3, nvec[2]);
   EXPECT_EQ(3u, nvec.size());

   EXPECT_EQ(-2, tvec[0]);
   EXPECT_EQ(-3, tvec[1]);
   EXPECT_EQ(-4, tvec[2]);
   EXPECT_EQ(3u, tvec.size());
}

TEST(VecOps, MathScalar)
{
   RVec<double> ref{1, 2, 3};
   RVec<double> v(ref);
   int scalar = 3;
   auto plus = v + scalar;
   auto minus = v - scalar;
   auto mult = v * scalar;
   auto div = v / scalar;

   CheckEqual(plus, ref + scalar);
   CheckEqual(minus, ref - scalar);
   CheckEqual(mult, ref * scalar);
   CheckEqual(div, ref / scalar);

   // The same with views
   RVec<double> w(ref.data(), ref.size());
   plus = w + scalar;
   minus = w - scalar;
   mult = w * scalar;
   div = w / scalar;

   CheckEqual(plus, ref + scalar);
   CheckEqual(minus, ref - scalar);
   CheckEqual(mult, ref * scalar);
   CheckEqual(div, ref / scalar);
}

TEST(VecOps, MathScalarInPlace)
{
   RVec<double> ref{1, 2, 3};
   const RVec<double> v(ref);
   int scalar = 3;
   auto plus = v;
   plus += scalar;
   auto minus = v;
   minus -= scalar;
   auto mult = v;
   mult *= scalar;
   auto div = v;
   div /= scalar;

   CheckEqual(plus, ref + scalar);
   CheckEqual(minus, ref - scalar);
   CheckEqual(mult, ref * scalar);
   CheckEqual(div, ref / scalar);
}

TEST(VecOps, MathVector)
{
   RVec<double> ref{1, 2, 3};
   RVec<double> vec{3, 4, 5};
   RVec<double> v(ref);
   auto plus = v + vec;
   auto minus = v - vec;
   auto mult = v * vec;
   auto div = v / vec;

   CheckEqual(plus, ref + vec);
   CheckEqual(minus, ref - vec);
   CheckEqual(mult, ref * vec);
   CheckEqual(div, ref / vec);

   // The same with 1 view
   RVec<double> w(ref.data(), ref.size());
   plus = w + vec;
   minus = w - vec;
   mult = w * vec;
   div = w / vec;

   CheckEqual(plus, ref + vec);
   CheckEqual(minus, ref - vec);
   CheckEqual(mult, ref * vec);
   CheckEqual(div, ref / vec);

   // The same with 2 views
   RVec<double> w2(ref.data(), ref.size());
   plus = w + w2;
   minus = w - w2;
   mult = w * w2;
   div = w / w2;

   CheckEqual(plus, ref + w2);
   CheckEqual(minus, ref - w2);
   CheckEqual(mult, ref * w2);
   CheckEqual(div, ref / w2);
}

TEST(VecOps, MathVectorInPlace)
{
   RVec<double> ref{1, 2, 3};
   RVec<double> vec{3, 4, 5};
   RVec<double> v(ref);
   auto plus = v;
   plus += vec;
   auto minus = v;
   minus -= vec;
   auto mult = v;
   mult *= vec;
   auto div = v;
   div /= vec;

   CheckEqual(plus, ref + vec);
   CheckEqual(minus, ref - vec);
   CheckEqual(mult, ref * vec);
   CheckEqual(div, ref / vec);
}

TEST(VecOps, Filter)
{
   RVec<int> v{0, 1, 2, 3, 4, 5};
   const std::vector<int> vEvenRef{0, 2, 4};
   const std::vector<int> vOddRef{1, 3, 5};
   auto vEven = v[v % 2 == 0];
   auto vOdd = v[v % 2 == 1];
   CheckEqual(vEven, vEvenRef, "Even check");
   CheckEqual(vOdd, vOddRef, "Odd check");

   // now with the helper function
   vEven = Filter(v, [](int i) { return 0 == i % 2; });
   vOdd = Filter(v, [](int i) { return 1 == i % 2; });
   CheckEqual(vEven, vEvenRef, "Even check");
   CheckEqual(vOdd, vOddRef, "Odd check");
}

template <typename T, typename V>
std::string PrintRVec(RVec<T> v, V w)
{
   std::stringstream ss;
   ss << v << " " << w << std::endl;
   ss << v + w << std::endl;
   ss << v - w << std::endl;
   ss << v * w << std::endl;
   ss << v / w << std::endl;
   ss << (v > w) << std::endl;
   ss << (v >= w) << std::endl;
   ss << (v == w) << std::endl;
   ss << (v <= w) << std::endl;
   ss << (v < w) << std::endl;
   ss << w + v << std::endl;
   ss << w - v << std::endl;
   ss << w * v << std::endl;
   ss << w / v << std::endl;
   ss << (w > v) << std::endl;
   ss << (w >= v) << std::endl;
   ss << (w == v) << std::endl;
   ss << (w <= v) << std::endl;
   ss << (w < v) << std::endl;
   return ss.str();
}

TEST(VecOps, PrintOps)
{
   RVec<int> ref{1, 2, 3};
   RVec<int> v(ref);

   auto ref0 = R"ref0({ 1, 2, 3 } 2
{ 3, 4, 5 }
{ -1, 0, 1 }
{ 2, 4, 6 }
{ 0.5, 1, 1.5 }
{ 0, 0, 1 }
{ 0, 1, 1 }
{ 0, 1, 0 }
{ 1, 1, 0 }
{ 1, 0, 0 }
{ 3, 4, 5 }
{ 1, 0, -1 }
{ 2, 4, 6 }
{ 2, 1, 0.666667 }
{ 1, 0, 0 }
{ 1, 1, 0 }
{ 0, 1, 0 }
{ 0, 1, 1 }
{ 0, 0, 1 }
)ref0";
   auto t0 = PrintRVec(v, 2.);
   EXPECT_STREQ(t0.c_str(), ref0);
   auto ref1 = R"ref1({ 1, 2, 3 } { 3, 4, 5 }
{ 4, 6, 8 }
{ -2, -2, -2 }
{ 3, 8, 15 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 1, 1, 1 }
{ 1, 1, 1 }
{ 4, 6, 8 }
{ 2, 2, 2 }
{ 3, 8, 15 }
{ 3, 2, 1 }
{ 1, 1, 1 }
{ 1, 1, 1 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
)ref1";
   auto t1 = PrintRVec(v, ref + 2);
   EXPECT_STREQ(t1.c_str(), ref1);

   RVec<int> w(ref.data(), ref.size());

   auto ref2 = R"ref2({ 1, 2, 3 } 2
{ 3, 4, 5 }
{ -1, 0, 1 }
{ 2, 4, 6 }
{ 0.5, 1, 1.5 }
{ 0, 0, 1 }
{ 0, 1, 1 }
{ 0, 1, 0 }
{ 1, 1, 0 }
{ 1, 0, 0 }
{ 3, 4, 5 }
{ 1, 0, -1 }
{ 2, 4, 6 }
{ 2, 1, 0.666667 }
{ 1, 0, 0 }
{ 1, 1, 0 }
{ 0, 1, 0 }
{ 0, 1, 1 }
{ 0, 0, 1 }
)ref2";
   auto t2 = PrintRVec(v, 2.);
   EXPECT_STREQ(t2.c_str(), ref2);

   auto ref3 = R"ref3({ 1, 2, 3 } { 3, 4, 5 }
{ 4, 6, 8 }
{ -2, -2, -2 }
{ 3, 8, 15 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 1, 1, 1 }
{ 1, 1, 1 }
{ 4, 6, 8 }
{ 2, 2, 2 }
{ 3, 8, 15 }
{ 3, 2, 1 }
{ 1, 1, 1 }
{ 1, 1, 1 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
)ref3";
   auto t3 = PrintRVec(v, ref + 2);
   EXPECT_STREQ(t3.c_str(), ref3);
}

#ifdef R__HAS_VDT
#include <vdt/vdtMath.h>
#endif

TEST(VecOps, MathFuncs)
{
   RVec<double> u{1, 1, 1};
   RVec<double> v{1, 2, 3};
   RVec<double> w{1, 4, 27};

   CheckEqual(pow(1,v), u, " error checking math function pow");
   CheckEqual(pow(v,1), v, " error checking math function pow");
   CheckEqual(pow(v,v), w, " error checking math function pow");

   CheckEqual(sqrt(v), Map(v, [](double x) { return std::sqrt(x); }), " error checking math function sqrt");
   CheckEqual(log(v), Map(v, [](double x) { return std::log(x); }), " error checking math function log");
   CheckEqual(sin(v), Map(v, [](double x) { return std::sin(x); }), " error checking math function sin");
   CheckEqual(cos(v), Map(v, [](double x) { return std::cos(x); }), " error checking math function cos");
   CheckEqual(tan(v), Map(v, [](double x) { return std::tan(x); }), " error checking math function tan");
   CheckEqual(atan(v), Map(v, [](double x) { return std::atan(x); }), " error checking math function atan");
   CheckEqual(sinh(v), Map(v, [](double x) { return std::sinh(x); }), " error checking math function sinh");
   CheckEqual(cosh(v), Map(v, [](double x) { return std::cosh(x); }), " error checking math function cosh");
   CheckEqual(tanh(v), Map(v, [](double x) { return std::tanh(x); }), " error checking math function tanh");
   CheckEqual(asinh(v), Map(v, [](double x) { return std::asinh(x); }), " error checking math function asinh");
   CheckEqual(acosh(v), Map(v, [](double x) { return std::acosh(x); }), " error checking math function acosh");
   v /= 10.;
   CheckEqual(asin(v), Map(v, [](double x) { return std::asin(x); }), " error checking math function asin");
   CheckEqual(acos(v), Map(v, [](double x) { return std::acos(x); }), " error checking math function acos");
   CheckEqual(atanh(v), Map(v, [](double x) { return std::atanh(x); }), " error checking math function atanh");

#ifdef R__HAS_VDT
   #define CHECK_VDT_FUNC(F) \
   CheckEqual(fast_##F(v), Map(v, [](double x) { return vdt::fast_##F(x); }), "error checking vdt function " #F);

   CHECK_VDT_FUNC(exp)
   CHECK_VDT_FUNC(log)

   CHECK_VDT_FUNC(sin)
   CHECK_VDT_FUNC(sin)
   CHECK_VDT_FUNC(cos)
   CHECK_VDT_FUNC(atan)
   CHECK_VDT_FUNC(acos)
   CHECK_VDT_FUNC(atan)
#endif
}

TEST(VecOps, PhysicsSelections)
{
   // We emulate 8 muons
   RVec<short> mu_charge{1, 1, -1, -1, -1, 1, 1, -1};
   RVec<float> mu_pt{56.f, 45.f, 32.f, 24.f, 12.f, 8.f, 7.f, 6.2f};
   RVec<float> mu_eta{3.1f, -.2f, -1.1f, 1.f, 4.1f, 1.6f, 2.4f, -.5f};

   // Pick the pt of the muons with a pt greater than 10, an eta between -2 and 2 and a negative charge
   // or the ones with a pt > 20, outside the eta range -2:2 and with positive charge
   auto goodMuons_pt = mu_pt[(mu_pt > 10.f && abs(mu_eta) <= 2.f && mu_charge == -1) ||
                             (mu_pt > 15.f && abs(mu_eta) > 2.f && mu_charge == 1)];
   RVec<float> goodMuons_pt_ref = {56.f, 32.f, 24.f};
   CheckEqual(goodMuons_pt, goodMuons_pt_ref, "Muons quality cut");
}

template<typename T0>
void CheckEq(const T0 &v, const T0 &ref)
{
   auto vsize = v.size();
   auto refsize = ref.size();
   EXPECT_EQ(vsize, refsize) << "Sizes are: " << vsize << " " << refsize << std::endl;
   for (auto i : ROOT::TSeqI(vsize)) {
      EXPECT_EQ(v[i], ref[i]) << "RVecs differ" << std::endl;
   }
}

TEST(VecOps, InputOutput)
{
   auto filename = "vecops_inputoutput.root";
   auto treename = "t";

   const RVec<double> dref {1., 2., 3.};
   const RVec<float> fref {1.f, 2.f, 3.f};
   const RVec<UInt_t> uiref {1, 2, 3};
   const RVec<ULong_t> ulref {1UL, 2UL, 3UL};
   const RVec<ULong64_t> ullref {1ULL, 2ULL, 3ULL};
   const RVec<UShort_t> usref {1, 2, 3};
   const RVec<UChar_t> ucref {1, 2, 3};
   const RVec<Int_t> iref {1, 2, 3};;
   const RVec<Long_t> lref {1UL, 2UL, 3UL};;
   const RVec<Long64_t> llref {1ULL, 2ULL, 3ULL};
   const RVec<Short_t> sref {1, 2, 3};
   const RVec<Char_t> cref {1, 2, 3};
   const RVec<bool> bref {true, false, true};

   {
      auto d = dref;
      auto f = fref;
      auto ui = uiref;
      auto ul = ulref;
      auto ull = ullref;
      auto us = usref;
      auto uc = ucref;
      auto i = iref;
      auto l = lref;
      auto ll = llref;
      auto s = sref;
      auto c = cref;
      auto b = bref;
      TFile file(filename, "RECREATE");
      TTree t(treename, treename);
      t.Branch("d", &d);
      t.Branch("f", &f);
      t.Branch("ui", &ui);
      t.Branch("ul", &ul);
      t.Branch("ull", &ull);
      t.Branch("us", &us);
      t.Branch("uc", &uc);
      t.Branch("i", &i);
      t.Branch("l", &l);
      t.Branch("ll", &ll);
      t.Branch("s", &s);
      t.Branch("c", &c);
      t.Branch("b", &b);
      t.Fill();
      t.Write();
   }

   auto d = new RVec<double>();
   auto f = new RVec<float>;
   auto ui = new RVec<UInt_t>();
   auto ul = new RVec<ULong_t>();
   auto ull = new RVec<ULong64_t>();
   auto us = new RVec<UShort_t>();
   auto uc = new RVec<UChar_t>();
   auto i = new RVec<Int_t>();
   auto l = new RVec<Long_t>();
   auto ll = new RVec<Long64_t>();
   auto s = new RVec<Short_t>();
   auto c = new RVec<Char_t>();
   auto b = new RVec<bool>();

   TFile file(filename);
   TTree *tp;
   file.GetObject(treename, tp);
   auto &t = *tp;

   t.SetBranchAddress("d", &d);
   t.SetBranchAddress("f", &f);
   t.SetBranchAddress("ui", &ui);
   t.SetBranchAddress("ul", &ul);
   t.SetBranchAddress("ull", &ull);
   t.SetBranchAddress("us", &us);
   t.SetBranchAddress("uc", &uc);
   t.SetBranchAddress("i", &i);
   t.SetBranchAddress("l", &l);
   t.SetBranchAddress("ll", &ll);
   t.SetBranchAddress("s", &s);
   t.SetBranchAddress("c", &c);
   t.SetBranchAddress("b", &b);

   t.GetEntry(0);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*b, bref);

   gSystem->Unlink(filename);

}

TEST(VecOps, SimpleStatOps)
{
   RVec<double> v0 {};
   ASSERT_DOUBLE_EQ(Sum(v0), 0.);
   ASSERT_DOUBLE_EQ(Mean(v0), 0.);
   ASSERT_DOUBLE_EQ(StdDev(v0), 0.);
   ASSERT_DOUBLE_EQ(Var(v0), 0.);

   RVec<double> v1 {42.};
   ASSERT_DOUBLE_EQ(Sum(v1), 42.);
   ASSERT_DOUBLE_EQ(Mean(v1), 42.);
   ASSERT_DOUBLE_EQ(Max(v1), 42.);
   ASSERT_DOUBLE_EQ(Min(v1), 42.);
   ASSERT_DOUBLE_EQ(ArgMax(v1), 0);
   ASSERT_DOUBLE_EQ(ArgMin(v1), 0);
   ASSERT_DOUBLE_EQ(StdDev(v1), 0.);
   ASSERT_DOUBLE_EQ(Var(v1), 0.);

   RVec<double> v2 {1., 2., 3.};
   ASSERT_DOUBLE_EQ(Sum(v2), 6.);
   ASSERT_DOUBLE_EQ(Mean(v2), 2.);
   ASSERT_DOUBLE_EQ(Max(v2), 3.);
   ASSERT_DOUBLE_EQ(Min(v2), 1.);
   ASSERT_DOUBLE_EQ(ArgMax(v2), 2);
   ASSERT_DOUBLE_EQ(ArgMin(v2), 0);
   ASSERT_DOUBLE_EQ(Var(v2), 1.);
   ASSERT_DOUBLE_EQ(StdDev(v2), 1.);

   RVec<double> v3 {10., 20., 32.};
   ASSERT_DOUBLE_EQ(Sum(v3), 62.);
   ASSERT_DOUBLE_EQ(Mean(v3), 20.666666666666668);
   ASSERT_DOUBLE_EQ(Max(v3), 32.);
   ASSERT_DOUBLE_EQ(Min(v3), 10.);
   ASSERT_DOUBLE_EQ(ArgMax(v3), 2);
   ASSERT_DOUBLE_EQ(ArgMin(v3), 0);
   ASSERT_DOUBLE_EQ(Var(v3), 121.33333333333337);
   ASSERT_DOUBLE_EQ(StdDev(v3), 11.015141094572206);

   RVec<int> v4 {2, 3, 1};
   ASSERT_DOUBLE_EQ(Sum(v4), 6.);
   ASSERT_DOUBLE_EQ(Mean(v4), 2.);
   ASSERT_DOUBLE_EQ(Max(v4), 3);
   ASSERT_DOUBLE_EQ(Min(v4), 1);
   ASSERT_DOUBLE_EQ(ArgMax(v4), 1);
   ASSERT_DOUBLE_EQ(ArgMin(v4), 2);
   ASSERT_DOUBLE_EQ(Var(v4), 1.);
   ASSERT_DOUBLE_EQ(StdDev(v4), 1.);

   RVec<int> v5 {2, 3, 1, 4};
   ASSERT_DOUBLE_EQ(Sum(v5), 10);
   ASSERT_DOUBLE_EQ(Mean(v5), 2.5);
   ASSERT_DOUBLE_EQ(Max(v5), 4);
   ASSERT_DOUBLE_EQ(Min(v5), 1);
   ASSERT_DOUBLE_EQ(ArgMax(v5), 3);
   ASSERT_DOUBLE_EQ(ArgMin(v5), 2);
   ASSERT_DOUBLE_EQ(Var(v5), 5./3);
   ASSERT_DOUBLE_EQ(StdDev(v5), std::sqrt(5./3));

   const ROOT::Math::PtEtaPhiMVector lv0 {15.5f, .3f, .1f, 105.65f};
   const ROOT::Math::PtEtaPhiMVector lv1 {34.32f, 2.2f, 3.02f, 105.65f};
   const ROOT::Math::PtEtaPhiMVector lv2 {12.95f, 1.32f, 2.2f, 105.65f};
   const ROOT::Math::PtEtaPhiMVector lv_sum_ref = lv0 + lv1 + lv2;
   const ROOT::Math::PtEtaPhiMVector lv_mean_ref = lv_sum_ref / 3;
   RVec<ROOT::Math::PtEtaPhiMVector> v6 {lv0, lv1, lv2};
   const ROOT::Math::PtEtaPhiMVector lv_sum = Sum(v6, ROOT::Math::PtEtaPhiMVector());
   const ROOT::Math::PtEtaPhiMVector lv_mean = Mean(v6, ROOT::Math::PtEtaPhiMVector());
   CheckEqual(lv_sum, lv_sum_ref);
   CheckEqual(lv_mean, lv_mean_ref);
}

TEST(VecOps, Any)
{
   RVec<int> vi {0, 1, 2};
   EXPECT_TRUE(Any(vi));
   vi = {0, 0, 0};
   EXPECT_FALSE(Any(vi));
   vi = {1, 1};
   EXPECT_TRUE(Any(vi));
}

TEST(VecOps, All)
{
   RVec<int> vi {0, 1, 2};
   EXPECT_FALSE(All(vi));
   vi = {0, 0, 0};
   EXPECT_FALSE(All(vi));
   vi = {1, 1};
   EXPECT_TRUE(All(vi));
}

TEST(VecOps, Argsort)
{
   RVec<int> v{2, 0, 1};
   using size_type = typename RVec<int>::size_type;
   auto i = Argsort(v);
   RVec<size_type> ref{1, 2, 0};
   CheckEqual(i, ref);
}

TEST(VecOps, ArgsortWithComparisonOperator)
{
   RVec<int> v{2, 0, 1};
   using size_type = typename RVec<int>::size_type;

   auto i1 = Argsort(v, [](int x, int y){ return x < y; });
   RVec<size_type> ref1{1, 2, 0};
   CheckEqual(i1, ref1);

   auto i2 = Argsort(v, [](int x, int y){ return x > y; });
   RVec<size_type> ref2{0, 2, 1};
   CheckEqual(i2, ref2);
}
TEST(VecOps, StableArgsort)
{
   RVec<int> v{2, 0, 2, 1};
   using size_type = typename RVec<int>::size_type;
   auto i = StableArgsort(v);
   RVec<size_type> ref{1, 3, 0, 2};
   CheckEqual(i, ref);

   // Test for stability
   RVec<int> v1{0, 0, 2, 2, 2, 2, 1, 2, 1, 0, 1, 0, 0, 2, 0, 0, 0, 1, 1, 2};
   auto i1 = StableArgsort(v1);
   RVec<size_type> ref1{0, 1, 9, 11, 12, 14, 15, 16, 6, 8, 10, 17, 18, 2, 3, 4, 5, 7, 13, 19};
   CheckEqual(i1, ref1);
}

TEST(VecOps, StableArgsortWithComparisonOperator)
{
   RVec<int> v{2, 0, 2, 1};
   using size_type = typename RVec<int>::size_type;

   auto i1 = Argsort(v, [](int x, int y) { return x < y; });
   RVec<size_type> ref1{1, 3, 0, 2};
   CheckEqual(i1, ref1);

   auto i2 = Argsort(v, [](int x, int y) { return x > y; });
   RVec<size_type> ref2{0, 2, 3, 1};
   CheckEqual(i2, ref2);
}

TEST(VecOps, TakeIndices)
{
   RVec<int> v0{2, 0, 1};
   RVec<typename RVec<int>::size_type> i{1, 2, 0, 0, 0};
   auto v1 = Take(v0, i);
   RVec<int> ref{0, 1, 2, 2, 2};
   CheckEqual(v1, ref);
}

TEST(VecOps, TakeFirst)
{
   RVec<int> v0{0, 1, 2};

   auto v1 = Take(v0, 2);
   RVec<int> ref{0, 1};
   CheckEqual(v1, ref);

   // Corner-case: Take zero entries
   auto v2 = Take(v0, 0);
   RVec<int> none{};
   CheckEqual(v2, none);
}

TEST(VecOps, TakeLast)
{
   RVec<int> v0{0, 1, 2};

   auto v1 = Take(v0, -2);
   RVec<int> ref{1, 2};
   CheckEqual(v1, ref);

   // Corner-case: Take zero entries
   auto v2 = Take(v0, 0);
   RVec<int> none{};
   CheckEqual(v2, none);
}

TEST(VecOps, Drop)
{
   RVec<int> v1{2, 0, 1};

   // also test that out-of-bound and repeated indices are ignored
   auto v2 = Drop(v1, {2, 2, 3});
   CheckEqual(v2, RVec<int>{2,0});

   EXPECT_EQ(Drop(RVec<int>{}, {1, 2, 3}).size(), 0u);
   EXPECT_EQ(Drop(RVec<int>{}, {}).size(), 0);
   CheckEqual(Drop(v1, {}), v1);
}

TEST(VecOps, Reverse)
{
   RVec<int> v0{0, 1, 2};

   auto v1 = Reverse(v0);
   RVec<int> ref{2, 1, 0};
   CheckEqual(v1, ref);

   // Corner-case: Empty vector
   RVec<int> none{};
   auto v2 = Reverse(none);
   CheckEqual(v2, none);
}

TEST(VecOps, Sort)
{
   RVec<int> v{2, 0, 1};

   // Sort in ascending order
   auto v1 = Sort(v);
   RVec<int> ref{0, 1, 2};
   CheckEqual(v1, ref);

   // Corner-case: Empty vector
   RVec<int> none{};
   auto v2 = Sort(none);
   CheckEqual(v2, none);
}

TEST(VecOps, SortWithComparisonOperator)
{
   RVec<int> v{2, 0, 1};

   // Sort with comparison operator
   auto v1 = Sort(v, std::greater<int>());
   RVec<int> ref{2, 1, 0};
   CheckEqual(v1, ref);

   // Corner-case: Empty vector
   RVec<int> none{};
   auto v2 = Sort(none, std::greater<int>());
   CheckEqual(v2, none);
}

TEST(VecOps, StableSort)
{
   RVec<int> v{2, 0, 2, 1};

   // Sort in ascending order
   auto v1 = StableSort(v);
   RVec<int> ref{0, 1, 2, 2};
   CheckEqual(v1, ref);

   // Corner-case: Empty vector
   RVec<int> none{};
   auto v2 = StableSort(none);
   CheckEqual(v2, none);
}

TEST(VecOps, StableSortWithComparisonOperator)
{
   RVec<int> v{2, 0, 2, 1};

   // Sort with comparison operator
   auto v1 = StableSort(v, std::greater<int>());
   RVec<int> ref{2, 2, 1, 0};
   CheckEqual(v1, ref);

   // Corner-case: Empty vector
   RVec<int> none{};
   auto v2 = StableSort(none, std::greater<int>());
   CheckEqual(v2, none);

   // Check stability
   RVec<RVec<int>> vv{{0, 2}, {2, 2}, {0, 2}, {0, 2}, {2, 0}, {0, 1}, {1, 0}, {1, 2}, {2, 0}, {0, 0},
                      {1, 1}, {0, 2}, {2, 1}, {2, 0}, {1, 1}, {1, 2}, {2, 2}, {1, 1}, {0, 2}, {0, 1}};
   RVec<RVec<int>> vv1Ref{{0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, {1, 0}, {1, 1},
                          {1, 1}, {1, 1}, {1, 2}, {1, 2}, {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}};
   auto vv1 = ROOT::VecOps::StableSort(
      ROOT::VecOps::StableSort(
         vv, [](const RVec<int> &vSub1, const RVec<int> &vSub2) -> bool { return vSub1[1] < vSub2[1]; }),
      [](const RVec<int> &vSub1, const RVec<int> &vSub2) -> bool { return vSub1[0] < vSub2[0]; });
   bool isVv1Correct =
      All(Map(vv1, vv1Ref, [](const RVec<int> &vSub, const RVec<int> &vRefSub) { return All(vSub == vRefSub); }));
   EXPECT_TRUE(isVv1Correct);
}

TEST(VecOps, RVecBool)
{
   // std::vector<bool> is special, so RVec<bool> is special
   RVec<bool> v{true, false};
   auto v2 = v;
   EXPECT_EQ(v[0], true);
   EXPECT_EQ(v[1], false);
   EXPECT_EQ(v.size(), 2u);
   CheckEqual(v2, v);
}

TEST(VecOps, CombinationsTwoVectors)
{
   RVec<int> v1{1, 2, 3};
   RVec<int> v2{-4, -5};

   auto idx = Combinations(v1, v2);
   auto c1 = Take(v1, idx[0]);
   auto c2 = Take(v2, idx[1]);
   auto v3 = c1 * c2;

   RVec<int> ref{-4, -5, -8, -10, -12, -15};
   CheckEqual(v3, ref);

   // Test overload with size as input
   auto idx3 = Combinations(v1.size(), v2.size());
   auto c3 = Take(v1, idx3[0]);
   auto c4 = Take(v2, idx3[1]);
   auto v4 = c3 * c4;

   CheckEqual(v4, ref);

   // Corner-case: One collection is empty
   RVec<int> empty_int{};
   auto idx2 = Combinations(v1, empty_int);
   RVec<size_t> empty_size{};
   CheckEqual(idx2[0], empty_size);
   CheckEqual(idx2[1], empty_size);
}

TEST(VecOps, UniqueCombinationsSingleVector)
{
   // Doubles: x + y
   RVec<int> v1{1, 2, 3};
   auto idx1 = Combinations(v1, 2);
   auto c1 = Take(v1, idx1[0]);
   auto c2 = Take(v1, idx1[1]);
   auto v2 = c1 + c2;
   RVec<int> ref1{
      3,  // 1+2
      4,  // 1+3
      5}; // 2+3
   CheckEqual(v2, ref1);

   // Triples: x * y * z
   RVec<int> v3{1, 2, 3, 4};
   auto idx2 = Combinations(v3, 3);
   auto c3 = Take(v3, idx2[0]);
   auto c4 = Take(v3, idx2[1]);
   auto c5 = Take(v3, idx2[2]);
   auto v4 = c3 * c4 * c5;
   RVec<int> ref2{
      6,  // 1*2*3
      8,  // 1*2*3
      12, // 1*3*4
      24};// 2*3*4
   CheckEqual(v4, ref2);

   // Corner-case: Single combination
   RVec<int> v5{1};
   auto idx3 = Combinations(v5, 1);
   EXPECT_EQ(idx3.size(), 1u);
   EXPECT_EQ(idx3[0].size(), 1u);
   EXPECT_EQ(idx3[0][0], 0u);

   // Corner-case: Insert empty vector
   RVec<int> empty_int{};
   auto idx4 = Combinations(empty_int, 0);
   EXPECT_EQ(idx4.size(), 0u);

   // Corner-case: Request "zero-tuples"
   auto idx5 = Combinations(v1, 0);
   EXPECT_EQ(idx5.size(), 0u);
}

TEST(VecOps, PrintCollOfNonPrintable)
{
   auto code = "class A{};ROOT::RVec<A> v(1);v";
   auto ret = gInterpreter->ProcessLine(code);
   EXPECT_TRUE(0 != ret) << "Error in printing an RVec collection of non printable objects.";
}

TEST(VecOps, Nonzero)
{
   RVec<int> v1{0, 1, 0, 3, 4, 0, 6};
   RVec<float> v2{0, 1, 0, 3, 4, 0, 6};
   auto v3 = Nonzero(v1);
   auto v4 = Nonzero(v2);
   RVec<size_t> ref1{1, 3, 4, 6};
   CheckEqual(v3, ref1);
   CheckEqual(v4, ref1);

   auto v5 = v1[v1<2];
   auto v6 = Nonzero(v5);
   RVec<size_t> ref2{1};
   CheckEqual(v6, ref2);
}

TEST(VecOps, Intersect)
{
   RVec<int> v1{0, 1, 2, 3};
   RVec<int> v2{2, 3, 4, 5};
   auto v3 = Intersect(v1, v2);
   RVec<int> ref1{2, 3};
   CheckEqual(v3, ref1);

   RVec<int> v4{4, 5, 3, 2};
   auto v5 = Intersect(v1, v4);
   CheckEqual(v5, ref1);

   // v2 already sorted
   auto v6 = Intersect(v1, v2, true);
   CheckEqual(v6, ref1);
}

TEST(VecOps, Where)
{
   // Use two vectors as arguments
   RVec<float> v0{1, 2, 3, 4};
   RVec<float> v1{-1, -2, -3, -4};
   auto v3 = Where(v0 > 1 && v0 < 4, v0, v1);
   RVec<float> ref1{-1, 2, 3, -4};
   CheckEqual(v3, ref1);

   // Broadcast false argument
   auto v4 = Where(v0 == 2 || v0 == 4, v0, -1.0f);
   RVec<float> ref2{-1, 2, -1, 4};
   CheckEqual(v4, ref2);

   // Broadcast true argument
   auto v5 = Where(v0 == 2 || v0 == 4, -1.0f, v1);
   RVec<float> ref3{-1, -1, -3, -1};
   CheckEqual(v5, ref3);

   // Broadcast both arguments
   auto v6 = Where(v0 == 2 || v0 == 4, -1.0f, 1.0f);
   RVec<float> ref4{1, -1, 1, -1};
   CheckEqual(v6, ref4);

   // Two temporary vectors as arguments
   auto v7 = Where(v0 > 1 && v0 < 4, RVecF{1, 2, 3, 4}, RVecF{-1, -2, -3, -4});
   RVecF ref5{-1, 2, 3, -4};
   CheckEqual(v3, ref1);

   // Scalar value of a different type
   auto v8 = Where(v0 > 1, v0, -1);
   CheckEqual(v8, RVecF{-1, 2, 3, 4});
   auto v9 = Where(v0 > 1, -1, v0);
   CheckEqual(v9, RVecF{1, -1, -1, -1});
}

TEST(VecOps, AtWithFallback)
{
   RVec<float> v({1.f, 2.f, 3.f});
   EXPECT_FLOAT_EQ(v.at(7, 99.f), 99.f);
}

TEST(VecOps, Concatenate)
{
   RVec<float> rvf {0.f, 1.f, 2.f};
   RVec<int> rvi {7, 8, 9};
   const auto res = Concatenate(rvf, rvi);
   const RVec<float> ref { 0.00000f, 1.00000f, 2.00000f, 7.00000f, 8.00000f, 9.00000f };
   CheckEqual(res, ref);
}

TEST(VecOps, DeltaPhi)
{
   // Two scalars (radians)
   // NOTE: These tests include the checks of the poundary effects
   const float c1 = M_PI;
   EXPECT_EQ(DeltaPhi(0.f, 2.f), 2.f);
   EXPECT_EQ(DeltaPhi(1.f, 0.f), -1.f);
   EXPECT_EQ(DeltaPhi(-0.5f, 0.5f), 1.f);
   EXPECT_EQ(DeltaPhi(0.f, 2.f * c1 - 1.f), -1.f);
   EXPECT_EQ(DeltaPhi(0.f, 4.f * c1 - 1.f), -1.f);
   EXPECT_EQ(DeltaPhi(0.f, -2.f * c1 + 1.f), 1.f);
   EXPECT_EQ(DeltaPhi(0.f, -4.f * c1 + 1.f), 1.f);

   // Two scalars (degrees)
   const float c2 = 180.f;
   EXPECT_EQ(DeltaPhi(0.f, 2.f, c2), 2.f);
   EXPECT_EQ(DeltaPhi(1.f, 0.f, c2), -1.f);
   EXPECT_EQ(DeltaPhi(-0.5f, 0.5f, c2), 1.f);
   EXPECT_EQ(DeltaPhi(0.f, 2.f * c2 - 1.f, c2), -1.f);
   EXPECT_EQ(DeltaPhi(0.f, 4.f * c2 - 1.f, c2), -1.f);
   EXPECT_EQ(DeltaPhi(0.f, -2.f * c2 + 1.f, c2), 1.f);
   EXPECT_EQ(DeltaPhi(0.f, -4.f * c2 + 1.f, c2), 1.f);

   // Two vectors
   RVec<float> v1 = {0.f, 1.f, -0.5f, 0.f, 0.f, 0.f, 0.f};
   RVec<float> v2 = {2.f, 0.f, 0.5f, 2.f * c1 - 1.f, 4.f * c1 - 1.f, -2.f * c1 + 1.f, -4.f * c1 + 1.f};
   auto dphi1 = DeltaPhi(v1, v2);
   RVec<float> r1 = {2.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f};
   CheckEqual(dphi1, r1);

   auto dphi2 = DeltaPhi(v2, v1);
   auto r2 = r1 * -1.f;
   CheckEqual(dphi2, r2);

   // Check against TLorentzVector
   for (std::size_t i = 0; i < v1.size(); i++) {
      TLorentzVector p1, p2;
      p1.SetPtEtaPhiM(1.f, 1.f, v1[i], 1.f);
      p2.SetPtEtaPhiM(1.f, 1.f, v2[i], 1.f);
      EXPECT_NEAR(float(p2.DeltaPhi(p1)), dphi1[i], 1e-6);
   }

   // Vector and scalar
   RVec<float> v3 = {0.f, 1.f, c1, c1 + 1.f, 2.f * c1, 2.f * c1 + 1.f, -1.f, -c1, -c1 + 1.f, -2.f * c1, -2.f * c1 + 1.f};
   float v4 = 1.f;
   auto dphi3 = DeltaPhi(v3, v4);
   RVec<float> r3 = {1.f, 0.f, 1.f - c1, c1, 1.f, 0.f, 2.f, -c1 + 1.f, c1, 1.f, 0.f};
   CheckEqual(dphi3, r3);

   auto dphi4 = DeltaPhi(v4, v3);
   auto r4 = -1.f * r3;
   CheckEqual(dphi4, r4);
}

TEST(VecOps, InvariantMass)
{
   // Dummy particle collections
   RVec<double> mass1 = {50,  50,  50,   50,   100};
   RVec<double> pt1 =   {0,   5,   5,    10,   10};
   RVec<double> eta1 =  {0.0, 0.0, -1.0, 0.5,  2.5};
   RVec<double> phi1 =  {0.0, 0.0, 0.0,  -0.5, -2.4};

   RVec<double> mass2 = {40,  40,  40,  40,  30};
   RVec<double> pt2 =   {0,   5,   5,   10,  2};
   RVec<double> eta2 =  {0.0, 0.0, 0.5, 0.4, 1.2};
   RVec<double> phi2 =  {0.0, 0.0, 0.0, 0.5, 2.4};

   // Compute invariant mass of two particle system using both collections
   const auto invMass = InvariantMasses(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2);

   for(size_t i=0; i<mass1.size(); i++) {
      TLorentzVector p1, p2;
      p1.SetPtEtaPhiM(pt1[i], eta1[i], phi1[i], mass1[i]);
      p2.SetPtEtaPhiM(pt2[i], eta2[i], phi2[i], mass2[i]);
      // NOTE: The accuracy of the optimized trigonometric functions is relatively
      // low and the test start to fail with an accuracy of 1e-5.
      EXPECT_NEAR((p1 + p2).M(), invMass[i], 1e-4);
   }

   // Compute invariant mass of multiple-particle system using a single collection
   const auto invMass2 = InvariantMass(pt1, eta1, phi1, mass1);

   TLorentzVector p3;
   p3.SetPtEtaPhiM(pt1[0], eta1[0], phi1[0], mass1[0]);
   for(size_t i=1; i<mass1.size(); i++) {
      TLorentzVector p4;
      p4.SetPtEtaPhiM(pt1[i], eta1[i], phi1[i], mass1[i]);
      p3 += p4;
   }

   EXPECT_NEAR(p3.M(), invMass2, 1e-4);

   const auto invMass3 = InvariantMass(pt2, eta2, phi2, mass2);

   TLorentzVector p5;
   p5.SetPtEtaPhiM(pt2[0], eta2[0], phi2[0], mass2[0]);
   for(size_t i=1; i<mass2.size(); i++) {
      TLorentzVector p6;
      p6.SetPtEtaPhiM(pt2[i], eta2[i], phi2[i], mass2[i]);
      p5 += p6;
   }

   EXPECT_NEAR(p5.M(), invMass3, 1e-4);
}

TEST(VecOps, DeltaR)
{
   RVec<double> eta1 =  {0.1, -1.0, -1.0, 0.5,  -2.5};
   RVec<double> eta2 =  {0.0, 0.0, 0.5, 2.4, 1.2};
   RVec<double> phi1 =  {1.0, 5.0, -1.0,  -0.5, -2.4};
   RVec<double> phi2 =  {0.0, 3.0, 6.0, 1.5, 1.4};

   auto dr = DeltaR(eta1, eta2, phi1, phi2);
   auto dr2 = DeltaR(eta2, eta1, phi2, phi1);

   for (std::size_t i = 0; i < eta1.size(); i++) {
      // Check against TLorentzVector
      TLorentzVector p1, p2;
      p1.SetPtEtaPhiM(1.f, eta1[i], phi1[i], 1.f);
      p2.SetPtEtaPhiM(1.f, eta2[i], phi2[i], 1.f);
      auto dr3 = p2.DeltaR(p1);
      EXPECT_NEAR(dr3, dr[i], 1e-6);
      EXPECT_NEAR(dr3, dr2[i], 1e-6);

      // Check scalar implementation
      auto dr4 = DeltaR(eta1[i], eta2[i], phi1[i], phi2[i]);
      EXPECT_NEAR(dr3, dr4, 1e-6);
   }
}

TEST(VecOps, Map)
{
   RVec<float> a({1.f, 2.f, 3.f});
   RVec<float> b({4.f, 5.f, 6.f});
   RVec<float> c({7.f, 8.f, 9.f});

   auto mod = [](float x, int y, double z) { return sqrt(x * x + y * y + z * z); };

   auto res = Map(a, c, c, mod);

   RVec<double> ref{9.9498743710661994, 11.489125293076057, 13.076696830622021};
   CheckEqual(res, ref);
}

TEST(VecOps, Construct)
{
   RVec<float> pts {15.5f, 34.32f, 12.95f};
   RVec<float> etas {.3f, 2.2f, 1.32f};
   RVec<float> phis {.1f, 3.02f, 2.2f};
   RVec<float> masses {105.65f, 105.65f, 105.65f};
   auto fourVects = Construct<ROOT::Math::PtEtaPhiMVector>(pts, etas, phis, masses);
   const ROOT::Math::PtEtaPhiMVector ref0 {15.5f, .3f, .1f, 105.65f};
   const ROOT::Math::PtEtaPhiMVector ref1 {34.32f, 2.2f, 3.02f, 105.65f};
   const ROOT::Math::PtEtaPhiMVector ref2 {12.95f, 1.32f, 2.2f, 105.65f};
   EXPECT_TRUE(fourVects[0] == ref0);
   EXPECT_TRUE(fourVects[1] == ref1);
   EXPECT_TRUE(fourVects[2] == ref2);
}

bool IsSmall(const RVec<int> &v)
{
   // the first array element is right after the 3 data members of SmallVectorBase
   return reinterpret_cast<std::uintptr_t>(v.begin()) - reinterpret_cast<std::uintptr_t>(&v) ==
          sizeof(void *) + 2 * sizeof(int);
}

// this is a regression test for https://github.com/root-project/root/issues/6796
TEST(VecOps, MemoryAdoptionAndClear)
{
   ROOT::RVec<int> v{1, 2, 3};
   EXPECT_TRUE(IsSmall(v));
   ROOT::RVec<int> v2(v.data(), v.size());
   v2[0] = 0;
   v2.clear();
   EXPECT_TRUE(All(v == RVec<int>{0, 2, 3}));
   v2.push_back(42);
   EXPECT_FALSE(IsSmall(v2)); // currently RVec does not go back to a small state after `clear()`
   EXPECT_TRUE(All(v2 == RVec<int>{42}));
}

// interaction between small buffer optimization and memory adoption
TEST(VecOps, MemoryAdoptionAndSBO)
{
   int *values = new int[3]{1, 2, 3};
   ROOT::RVec<int> v(values, 3);
   auto check = [](const RVec<int> &mv) {
      EXPECT_EQ(mv.size(), 3u);
      EXPECT_EQ(mv[0], 1);
      EXPECT_EQ(mv[1], 2);
      EXPECT_EQ(mv[2], 3);
   };
   check(v);
   EXPECT_FALSE(IsSmall(v));
   ROOT::RVec<int> v2 = std::move(v);
   EXPECT_TRUE(v.empty());
   check(v2);
   // this changes the RVec from memory adoption mode to "long" mode, even if the size is small
   // currently we don't allow going from memory adoption to small buffer mode directly, it could be future optimization
   v2.push_back(4);
   EXPECT_FALSE(IsSmall(v2));
   v2.clear();
   v2.push_back(1);
   v2.push_back(2);
   v2.push_back(3);
   check(v2);
   delete[] values;
   check(v2);
}

struct ThrowingCtor {
   ThrowingCtor() { throw std::runtime_error("This exception should have been caught."); }
};

struct ThrowingMove {
   ThrowingMove() = default;
   ThrowingMove(const ThrowingMove &) = default;
   ThrowingMove &operator=(const ThrowingMove &) = default;
   ThrowingMove(ThrowingMove &&) { throw std::runtime_error("This exception should have been caught."); }
   ThrowingMove &operator=(ThrowingMove &&) { throw std::runtime_error("This exception should have been caught."); }
};

struct ThrowingCopy {
   ThrowingCopy() = default;
   ThrowingCopy(const ThrowingCopy &) { throw std::runtime_error("This exception should have been caught."); }
   ThrowingCopy &operator=(const ThrowingCopy &)
   {
      throw std::runtime_error("This exception should have been caught.");
   }
   ThrowingCopy(ThrowingCopy &&) = default;
   ThrowingCopy &operator=(ThrowingCopy &&) = default;
};

// RVec does not guarantee exception safety, but we still want to test
// that we don't segfault or otherwise crash if element construction or move throws.
TEST(VecOps, NoExceptionSafety)
{
   EXPECT_NO_THROW(ROOT::RVec<ThrowingCtor>());

   EXPECT_THROW(ROOT::RVec<ThrowingCtor>(1), std::runtime_error);
   EXPECT_THROW(ROOT::RVec<ThrowingCtor>(42), std::runtime_error);
   ROOT::RVec<ThrowingCtor> v1;
   EXPECT_THROW(v1.push_back(ThrowingCtor{}), std::runtime_error);
   ROOT::RVec<ThrowingCtor> v2;
   EXPECT_THROW(v2.emplace_back(ThrowingCtor{}), std::runtime_error);

   ROOT::RVec<ThrowingMove> v3(2);
   ROOT::RVec<ThrowingMove> v4(42);
   EXPECT_THROW(std::swap(v3, v4), std::runtime_error);
   ThrowingMove tm;
   EXPECT_THROW(v3.emplace_back(std::move(tm)), std::runtime_error);

   ROOT::RVec<ThrowingCopy> v7;
   EXPECT_THROW(std::fill_n(std::back_inserter(v7), 16, ThrowingCopy{}), std::runtime_error);

   // now with memory adoption
   ThrowingCtor *p1 = new ThrowingCtor[0];
   ROOT::RVec<ThrowingCtor> v8(p1, 0);
   EXPECT_THROW(v8.push_back(ThrowingCtor{}), std::runtime_error);
   delete[] p1;

   ThrowingMove *p2 = new ThrowingMove[2];
   ROOT::RVec<ThrowingMove> v9(p2, 2);
   EXPECT_THROW(std::swap(v9, v3), std::runtime_error);
   delete[] p2;

   ThrowingCopy *p3 = new ThrowingCopy[2];
   ROOT::RVec<ThrowingCopy> v10(p3, 2);
   EXPECT_THROW(v10.push_back(*p3), std::runtime_error);
}

/*
Possible combinations of vectors to swap:
1. small <-> small
2. regular <-> regular (not small, not adopting)
3. adopting <-> adopting
4. small <-> regular (and vice versa)
5. small <-> adopting (and vice versa)
6. regular <-> adopting (and vice versa)
*/

TEST(VecOpsSwap, BothSmallVectors)
{
   RVec<int> fixed_vempty{};
   RVec<int> fixed_vshort1{1, 2, 3};
   RVec<int> fixed_vshort2{4, 5, 6};
   RVec<int> fixed_vshort3{7, 8};

   RVec<int> vempty{};
   RVec<int> vshort1{1, 2, 3};
   RVec<int> vshort2{4, 5, 6};
   RVec<int> vshort3{7, 8};

   swap(vshort1, vshort2); // swap of equal sizes

   CheckEqual(vshort1, fixed_vshort2);
   CheckEqual(vshort2, fixed_vshort1);
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vshort1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vshort2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vshort1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vshort2));

   swap(vshort1, vshort3); // left vector has bigger size

   CheckEqual(vshort1, fixed_vshort3);
   CheckEqual(vshort3, fixed_vshort2);
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vshort1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vshort3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vshort1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vshort3));

   swap(vshort1, vshort3); // left vector has smaller size

   CheckEqual(vshort1, fixed_vshort2);
   CheckEqual(vshort3, fixed_vshort3);
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vshort1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vshort3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vshort1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vshort3));

   swap(vempty, vshort2); // handling empty vectors

   CheckEqual(vempty, fixed_vshort1);
   CheckEqual(vshort2, fixed_vempty);
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vempty));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vshort2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vempty));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vshort2));
}

TEST(VecOpsSwap, BothRegularVectors)
{
   RVec<int> fixed_vreg1{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
   RVec<int> fixed_vreg2{4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
   RVec<int> fixed_vreg3{7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7};
   RVec<int> fixed_vmocksmall{0, 7};

   RVec<int> vreg1{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
   RVec<int> vreg2{4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
   RVec<int> vreg3{7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7};
   RVec<int> vmocksmall{0, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 7, 8, 9, 7, 8, 9};
   vmocksmall.erase(vmocksmall.begin() + 2, vmocksmall.end());
   // vmocksmall is a regular vector of size 2

   // verify that initally vectors are not small
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vmocksmall));

   swap(vreg1, vreg2); // swap of equal sizes

   CheckEqual(vreg1, fixed_vreg2);
   CheckEqual(vreg2, fixed_vreg1);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg2));

   swap(vreg1, vreg3); // left vector has bigger size

   CheckEqual(vreg1, fixed_vreg3);
   CheckEqual(vreg3, fixed_vreg2);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg3));

   swap(vreg1, vreg3); // left vector has smaller size

   CheckEqual(vreg1, fixed_vreg2);
   CheckEqual(vreg3, fixed_vreg3);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg3));

   swap(vreg3, vmocksmall); // handling artificially shortened regular vector as right argument

   CheckEqual(vreg3, fixed_vmocksmall);
   CheckEqual(vmocksmall, fixed_vreg3);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vmocksmall));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vmocksmall));

   swap(vreg3, vmocksmall); // handling artificially shortened regular vector as left argument

   CheckEqual(vreg3, fixed_vreg3);
   CheckEqual(vmocksmall, fixed_vmocksmall);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vmocksmall));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vmocksmall));
}

TEST(VecOpsSwap, BothAdoptingVectors)
{
   std::vector<int> v1{1, 2, 3};
   std::vector<int> v2{4, 5, 6};
   std::vector<int> v3{7};

   RVec<int> vadopt1(v1.data(), v1.size());
   RVec<int> vadopt2(v2.data(), v2.size());
   RVec<int> vadopt3(v3.data(), v3.size());

   swap(vadopt1, vadopt2); // swap of equal sizes

   CheckEqual(vadopt1, v2);
   CheckEqual(vadopt2, v1);
   EXPECT_EQ(&vadopt1[0], &v2[0]);
   EXPECT_EQ(&vadopt2[0], &v1[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt2));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt2));

   // check that adoption works in both directions
   v1[0] = 8;      // v1 is now adopted by vadopt2
   vadopt1[0] = 9; // vadopt1 adopts v2

   CheckEqual(vadopt1, v2);
   CheckEqual(vadopt2, v1);
   EXPECT_EQ(&vadopt1[0], &v2[0]);
   EXPECT_EQ(&vadopt2[0], &v1[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt2));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt2));

   swap(vadopt1, vadopt3); // left vector has bigger size

   CheckEqual(vadopt1, v3);
   CheckEqual(vadopt3, v2);
   EXPECT_EQ(&vadopt1[0], &v3[0]);
   EXPECT_EQ(&vadopt3[0], &v2[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt3));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt3));

   swap(vadopt1, vadopt3); // left vector has smaller size

   CheckEqual(vadopt1, v2);
   CheckEqual(vadopt3, v3);
   EXPECT_EQ(&vadopt1[0], &v2[0]);
   EXPECT_EQ(&vadopt3[0], &v3[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt3));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt3));
}

TEST(VecOpsSwap, SmallRegularVectors)
{
   RVec<int> fixed_vsmall{1, 2, 3};
   RVec<int> fixed_vreg1{4, 5, 6};
   RVec<int> fixed_vreg2{7, 8};
   RVec<int> fixed_vreg3{9, 10, 11, 12, 13, 14};
   RVec<int> fixed_vreg4{15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

   // need multiple hard copies since after swap of a small and a regular,
   // there is no fixed policy whether 2 regular vectors are produced or 1 small and 1 regular
   // currently a small and a regular vector are produced (this might change)
   RVec<int> vsmall1{1, 2, 3};
   RVec<int> vsmall2{1, 2, 3};
   RVec<int> vsmall3{1, 2, 3};
   RVec<int> vsmall4{1, 2, 3};
   RVec<int> vsmall5{1, 2, 3};
   RVec<int> vsmall6{1, 2, 3};
   RVec<int> vsmall7{1, 2, 3};

   RVec<int> vreg1{4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
   vreg1.erase(vreg1.begin() + 3, vreg1.end()); // regular vector of size 3
   RVec<int> vreg2{7, 8, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
   vreg2.erase(vreg2.begin() + 2, vreg2.end()); // regular vector of size 2
   RVec<int> vreg20{7, 8, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
   vreg20.erase(vreg20.begin() + 2, vreg20.end()); // regular vector of size 2
   RVec<int> vreg3{9, 10, 11, 12, 13, 14, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
   vreg3.erase(vreg3.begin() + 6, vreg3.end()); // regular vector of size 6
   RVec<int> vreg30{9, 10, 11, 12, 13, 14, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
   vreg30.erase(vreg30.begin() + 6, vreg30.end()); // regular vector of size 6
   RVec<int> vreg4{15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
   // vreg4 is a regular vector that cannot "fit" to small vector

   // verify that initally vectors are not small
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg4));

   swap(vsmall1, vreg1); // small <-> regular (same size)

   CheckEqual(vsmall1, fixed_vreg1);
   CheckEqual(vreg1, fixed_vsmall);
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vsmall1)); // the initially small vector remained small
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg1));  // the initially regular vector remained regular
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg1));

   swap(vreg1, vsmall1); // regular <-> small (same size)

   CheckEqual(vreg1, fixed_vreg1);
   CheckEqual(vsmall1, fixed_vsmall);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg1));  // the initially regular vector remained regular
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vsmall1)); // the initially small vector remained small
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall1));

   swap(vsmall3, vreg2); // longer small <-> shorter regular

   CheckEqual(vsmall3, fixed_vreg2);
   CheckEqual(vreg2, fixed_vsmall);
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vsmall3)); // the initially small vector remained small
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg2));  // the initially regular vector remained regular
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg2));

   swap(vreg20, vsmall4); // shorter regular <-> longer small

   CheckEqual(vreg20, fixed_vsmall);
   CheckEqual(vsmall4, fixed_vreg2);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg20)); // the initially regular vector remained regular
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vsmall4)); // the initially small vector remained small
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg20));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall4));

   swap(vsmall5, vreg3); // shorter small <-> longer regular

   CheckEqual(vsmall5, fixed_vreg3);
   CheckEqual(vreg3, fixed_vsmall);
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vsmall5)); // the initially small vector remained small
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg3));  // the initially regular vector remained regular
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall5));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg3));

   swap(vreg30, vsmall6); // shorter regular <-> longer small

   CheckEqual(vreg30, fixed_vsmall);
   CheckEqual(vsmall6, fixed_vreg3);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg30)); // the initially regular vector remained regular
   EXPECT_TRUE(ROOT::Detail::VecOps::IsSmall(vsmall6)); // the initially small vector remained small
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg30));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall6));

   swap(vsmall2, vreg4); // small <-> very long regular

   CheckEqual(vsmall2, fixed_vreg4);
   CheckEqual(vreg4, fixed_vsmall);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall2)); // the initially small vector is now regular
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg4));   // the initially regular vector remained regular
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg4));

   swap(vsmall2, vsmall7); // very long regular <-> small
   // vsmall2 is already swapped

   CheckEqual(vsmall2, fixed_vsmall);
   CheckEqual(vsmall7, fixed_vreg4);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall2)); // the initially regular vector remained regular
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vreg4));   // the initially small vector is now regular
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vreg4));
}

TEST(VecOpsSwap, SmallAdoptingVectors)
{
   RVec<int> fixed_vsmall{1, 2, 3};
   std::vector<int> v1{4, 5, 6};
   std::vector<int> v2{7, 8};
   std::vector<int> v3{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

   // need multiple hard copies since after swap of a small and an adopting,
   // an adopting and regular vector are produced
   RVec<int> vsmall1{1, 2, 3};
   RVec<int> vsmall2{1, 2, 3};
   RVec<int> vsmall3{1, 2, 3};
   RVec<int> vsmall4{1, 2, 3};
   RVec<int> vsmall5{1, 2, 3};
   RVec<int> vsmall6{1, 2, 3};

   RVec<int> vadopt1(v1.data(), v1.size());
   RVec<int> vadopt2(v2.data(), v2.size());
   RVec<int> vadopt3(v3.data(), v3.size());

   swap(vsmall1, vadopt1); // non-adopting <-> adopting (same size)

   CheckEqual(vsmall1, v1);
   CheckEqual(vadopt1, fixed_vsmall);
   EXPECT_EQ(&vsmall1[0], &v1[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vsmall1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vadopt1));

   swap(vsmall1, vsmall2); // adopting <-> non-adopting (same size)
   // vsmall1 is already swapped

   CheckEqual(vsmall1, fixed_vsmall);
   CheckEqual(vsmall2, v1);
   EXPECT_EQ(&vsmall2[0], &v1[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vsmall2));

   swap(vsmall3, vadopt2); // longer non-adopting <-> shorter adopting

   CheckEqual(vsmall3, v2);
   CheckEqual(vadopt2, fixed_vsmall);
   EXPECT_EQ(&vsmall3[0], &v2[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt2));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vsmall3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vadopt2));

   swap(vsmall3, vsmall4); // shorter adopting <-> longer non-adopting

   CheckEqual(vsmall3, fixed_vsmall);
   CheckEqual(vsmall4, v2);
   EXPECT_EQ(&vsmall4[0], &v2[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall4));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall3));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vsmall4));

   swap(vsmall5, vadopt3); // shorter non-adopting <-> longer adopting

   CheckEqual(vsmall5, v3);
   CheckEqual(vadopt3, fixed_vsmall);
   EXPECT_EQ(&vsmall5[0], &v3[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall5));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt3));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vsmall5));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vadopt3));

   swap(vsmall5, vsmall6); // longer adopting <-> shorter non-adopting

   CheckEqual(vsmall5, fixed_vsmall);
   CheckEqual(vsmall6, v3);
   EXPECT_EQ(&vsmall6[0], &v3[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall5));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vsmall6));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vsmall5));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vsmall6));
}

TEST(VecOpsSwap, RegularAdoptingVectors)
{
   RVec<int> fixed_vregular{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
   std::vector<int> v1{15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
   std::vector<int> v2{29};
   std::vector<int> v3{30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46};

   RVec<int> vregular{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
   RVec<int> vadopt1(v1.data(), v1.size());
   RVec<int> vadopt2(v2.data(), v2.size());
   RVec<int> vadopt3(v3.data(), v3.size());

   swap(vregular, vadopt1); // non-adopting <-> adopting (same size)

   CheckEqual(vregular, v1);
   CheckEqual(vadopt1, fixed_vregular);
   EXPECT_EQ(&vregular[0], &v1[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt1));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vadopt1));

   swap(vregular, vadopt1); // adopting <-> non-adopting (same size)

   CheckEqual(vregular, fixed_vregular);
   CheckEqual(vadopt1, v1);
   EXPECT_EQ(&vadopt1[0], &v1[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt1));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vregular));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt1));

   swap(vregular, vadopt2); // longer non-adopting <-> shorter adopting

   CheckEqual(vregular, v2);
   CheckEqual(vadopt2, fixed_vregular);
   EXPECT_EQ(&vregular[0], &v2[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt2));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vadopt2));

   swap(vregular, vadopt2); // shorter adopting <-> longer non-adopting

   CheckEqual(vregular, fixed_vregular);
   CheckEqual(vadopt2, v2);
   EXPECT_EQ(&vadopt2[0], &v2[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt2));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vregular));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt2));

   swap(vregular, vadopt3); // shorter non-adopting <-> longer adopting

   CheckEqual(vregular, v3);
   CheckEqual(vadopt3, fixed_vregular);
   EXPECT_EQ(&vregular[0], &v3[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt3));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vadopt3));

   swap(vregular, vadopt3); // longer adopting <-> shorter non-adopting

   CheckEqual(vregular, fixed_vregular);
   CheckEqual(vadopt3, v3);
   EXPECT_EQ(&vadopt3[0], &v3[0]);
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vregular));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsSmall(vadopt3));
   EXPECT_FALSE(ROOT::Detail::VecOps::IsAdopting(vregular));
   EXPECT_TRUE(ROOT::Detail::VecOps::IsAdopting(vadopt3));
}
