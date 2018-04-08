#include <gtest/gtest.h>
#include <ROOT/TVec.hxx>
#include <ROOT/TSeq.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <vector>
#include <sstream>

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
   ROOT::Experimental::VecOps::TVec<int> v;
   EXPECT_EQ(v.size(), 0u);
}

TEST(VecOps, InitListCtor)
{
   ROOT::Experimental::VecOps::TVec<int> v{1, 2, 3};
   EXPECT_EQ(v.size(), 3u);
   EXPECT_EQ(v[0], 1);
   EXPECT_EQ(v[1], 2);
   EXPECT_EQ(v[2], 3);
}

TEST(VecOps, CopyCtor)
{
   ROOT::Experimental::VecOps::TVec<int> v1{1, 2, 3};
   ROOT::Experimental::VecOps::TVec<int> v2(v1);
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

TEST(VecOps, CopyCtorCheckNoLeak)
{
   ROOT::Experimental::VecOps::TVec<TLeakChecker> ref;
   ref.emplace_back(TLeakChecker());
   ROOT::Experimental::VecOps::TVec<TLeakChecker> proxy(ref.data(), ref.size());
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
   ROOT::Experimental::VecOps::TVec<int> v1{1, 2, 3};
   ROOT::Experimental::VecOps::TVec<int> v2(std::move(v1));
   EXPECT_EQ(v1.size(), 0u);
   EXPECT_EQ(v2.size(), 3u);
}

TEST(VecOps, MathScalar)
{
   ROOT::Experimental::VecOps::TVec<double> ref{1, 2, 3};
   ROOT::Experimental::VecOps::TVec<double> v(ref);
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
   ROOT::Experimental::VecOps::TVec<double> w(ref.data(), ref.size());
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
   ROOT::Experimental::VecOps::TVec<double> ref{1, 2, 3};
   const ROOT::Experimental::VecOps::TVec<double> v(ref);
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
   ROOT::Experimental::VecOps::TVec<double> ref{1, 2, 3};
   ROOT::Experimental::VecOps::TVec<double> vec{3, 4, 5};
   ROOT::Experimental::VecOps::TVec<double> v(ref);
   auto plus = v + vec;
   auto minus = v - vec;
   auto mult = v * vec;
   auto div = v / vec;

   CheckEqual(plus, ref + vec);
   CheckEqual(minus, ref - vec);
   CheckEqual(mult, ref * vec);
   CheckEqual(div, ref / vec);

   // The same with 1 view
   ROOT::Experimental::VecOps::TVec<double> w(ref.data(), ref.size());
   plus = w + vec;
   minus = w - vec;
   mult = w * vec;
   div = w / vec;

   CheckEqual(plus, ref + vec);
   CheckEqual(minus, ref - vec);
   CheckEqual(mult, ref * vec);
   CheckEqual(div, ref / vec);

   // The same with 2 views
   ROOT::Experimental::VecOps::TVec<double> w2(ref.data(), ref.size());
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
   ROOT::Experimental::VecOps::TVec<double> ref{1, 2, 3};
   ROOT::Experimental::VecOps::TVec<double> vec{3, 4, 5};
   ROOT::Experimental::VecOps::TVec<double> v(ref);
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
   ROOT::Experimental::VecOps::TVec<int> v{0, 1, 2, 3, 4, 5};
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
std::string PrintTVec(ROOT::Experimental::VecOps::TVec<T> v, V w)
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
   ROOT::Experimental::VecOps::TVec<int> ref{1, 2, 3};
   ROOT::Experimental::VecOps::TVec<int> v(ref);

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
   auto t0 = PrintTVec(v, 2.);
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
   auto t1 = PrintTVec(v, ref + 2);
   EXPECT_STREQ(t1.c_str(), ref1);

   ROOT::Experimental::VecOps::TVec<int> w(ref.data(), ref.size());

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
   auto t2 = PrintTVec(v, 2.);
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
   auto t3 = PrintTVec(v, ref + 2);
   EXPECT_STREQ(t3.c_str(), ref3);
}

TEST(VecOps, MathFuncs)
{
   ROOT::Experimental::VecOps::TVec<double> v{1, 2, 3};
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
}

TEST(VecOps, PhysicsSelections)
{
   // We emulate 8 muons
   ROOT::Experimental::VecOps::TVec<short> mu_charge{1, 1, -1, -1, -1, 1, 1, -1};
   ROOT::Experimental::VecOps::TVec<float> mu_pt{56.f, 45.f, 32.f, 24.f, 12.f, 8.f, 7.f, 6.2f};
   ROOT::Experimental::VecOps::TVec<float> mu_eta{3.1f, -.2f, -1.1f, 1.f, 4.1f, 1.6f, 2.4f, -.5f};

   // Pick the pt of the muons with a pt greater than 10, an eta between -2 and 2 and a negative charge
   // or the ones with a pt > 20, outside the eta range -2:2 and with positive charge
   auto goodMuons_pt = mu_pt[(mu_pt > 10.f && abs(mu_eta) <= 2.f && mu_charge == -1) ||
                             (mu_pt > 15.f && abs(mu_eta) > 2.f && mu_charge == 1)];
   ROOT::Experimental::VecOps::TVec<float> goodMuons_pt_ref = {56.f, 32.f, 24.f};
   CheckEqual(goodMuons_pt, goodMuons_pt_ref, "Muons quality cut");
}

template<typename T0>
void CheckEq(const T0 &v, const T0 &ref)
{
   auto vsize = v.size();
   auto refsize = ref.size();
   EXPECT_EQ(vsize, refsize) << "Sizes are: " << vsize << " " << refsize << std::endl;
   for (auto i : ROOT::TSeqI(vsize)) {
      EXPECT_EQ(v[i], ref[i]) << "TVecs differ" << std::endl;
   }
}

TEST(VecOps, inputOutput)
{
   auto filename = "vecops_inputoutput.root";
   auto treename = "t";

   const ROOT::Experimental::VecOps::TVec<double>::Impl_t dref {1., 2., 3.};
   const ROOT::Experimental::VecOps::TVec<float>::Impl_t fref {1.f, 2.f, 3.f};
   const ROOT::Experimental::VecOps::TVec<UInt_t>::Impl_t uiref {1, 2, 3};
   const ROOT::Experimental::VecOps::TVec<ULong_t>::Impl_t ulref {1UL, 2UL, 3UL};
   const ROOT::Experimental::VecOps::TVec<ULong64_t>::Impl_t ullref {1ULL, 2ULL, 3ULL};
   const ROOT::Experimental::VecOps::TVec<UShort_t>::Impl_t usref {1, 2, 3};
   const ROOT::Experimental::VecOps::TVec<UChar_t>::Impl_t ucref {1, 2, 3};
   const ROOT::Experimental::VecOps::TVec<Int_t>::Impl_t iref {1, 2, 3};;
   const ROOT::Experimental::VecOps::TVec<Long_t>::Impl_t lref {1UL, 2UL, 3UL};;
   const ROOT::Experimental::VecOps::TVec<Long64_t>::Impl_t llref {1ULL, 2ULL, 3ULL};
   const ROOT::Experimental::VecOps::TVec<Short_t>::Impl_t sref {1, 2, 3};
   const ROOT::Experimental::VecOps::TVec<Char_t>::Impl_t cref {1, 2, 3};

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
      t.Fill();
      t.Write();
   }

   auto d = new ROOT::Experimental::VecOps::TVec<double>::Impl_t();
   auto f = new ROOT::Experimental::VecOps::TVec<float>::Impl_t;
   auto ui = new ROOT::Experimental::VecOps::TVec<UInt_t>::Impl_t();
   auto ul = new ROOT::Experimental::VecOps::TVec<ULong_t>::Impl_t();
   auto ull = new ROOT::Experimental::VecOps::TVec<ULong64_t>::Impl_t();
   auto us = new ROOT::Experimental::VecOps::TVec<UShort_t>::Impl_t();
   auto uc = new ROOT::Experimental::VecOps::TVec<UChar_t>::Impl_t();
   auto i = new ROOT::Experimental::VecOps::TVec<Int_t>::Impl_t();
   auto l = new ROOT::Experimental::VecOps::TVec<Long_t>::Impl_t();
   auto ll = new ROOT::Experimental::VecOps::TVec<Long64_t>::Impl_t();
   auto s = new ROOT::Experimental::VecOps::TVec<Short_t>::Impl_t();
   auto c = new ROOT::Experimental::VecOps::TVec<Char_t>::Impl_t();

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

   gSystem->Unlink(filename);

}

TEST(VecOps, SimpleStatOps)
{
   ROOT::Experimental::VecOps::TVec<double> v0 {};
   ASSERT_DOUBLE_EQ(Sum(v0), 0.);
   ASSERT_DOUBLE_EQ(Mean(v0), 0.);
   ASSERT_DOUBLE_EQ(StdDev(v0), 0.);
   ASSERT_DOUBLE_EQ(Var(v0), 0.);

   ROOT::Experimental::VecOps::TVec<double> v1 {42.};
   ASSERT_DOUBLE_EQ(Sum(v1), 42.);
   ASSERT_DOUBLE_EQ(Mean(v1), 42.);
   ASSERT_DOUBLE_EQ(StdDev(v1), 0.);
   ASSERT_DOUBLE_EQ(Var(v1), 0.);

   ROOT::Experimental::VecOps::TVec<double> v2 {1., 2., 3.};
   ASSERT_DOUBLE_EQ(Sum(v2), 6.);
   ASSERT_DOUBLE_EQ(Mean(v2), 2.);
   ASSERT_DOUBLE_EQ(Var(v2), 1.);
   ASSERT_DOUBLE_EQ(StdDev(v2), 1.);
   
   ROOT::Experimental::VecOps::TVec<double> v3 {10., 20., 32.};
   ASSERT_DOUBLE_EQ(Sum(v3), 62.);
   ASSERT_DOUBLE_EQ(Mean(v3), 20.666666666666668);
   ASSERT_DOUBLE_EQ(Var(v3), 121.33333333333337);
   ASSERT_DOUBLE_EQ(StdDev(v3), 11.015141094572206);
}
