#include "ROOT/StringUtils.hxx"

#include "gtest/gtest.h"

TEST(StringUtils, Split)
{
   // Test that ROOT::Split behaves like str.split from Python.

   auto test = [](std::string_view in, std::vector<std::string> const &ref, bool skipEmpty) {
      auto out = ROOT::Split(in, ",", skipEmpty);
      EXPECT_EQ(out, ref) << "ROOT::Split(\"" << in << "\") gave wrong result.";
   };
   test("a,b,c", {"a", "b", "c"}, false);
   test("a,b,c", {"a", "b", "c"}, true);
   test("a,,c", {"a", "", "c"}, false);
   test("a,,c", {"a", "c"}, true);
   test("a,,,", {"a", "", "", ""}, false);
   test(",,a,,,", {"", "", "a", "", "", ""}, false);
   test(",,,", {"", "", "", ""}, false);
   test(",,,", {}, true);
   test(",,a", {"", "", "a"}, false);
   test("", {""}, false);
}

TEST(StringUtils, Join)
{
   // Test that ROOT::Join behaves like str.join from Python.

   auto test = [](const std::string &ref, const std::string &sep, const std::vector<std::string> &strings) {
      auto out = ROOT::Join(sep, strings);
      EXPECT_EQ(out, ref) << "ROOT::Join gave wrong result.";
   };
   test("apple,orange,banana", ",", {"apple", "orange", "banana"});
   test("apple.orange.banana", ".", {"apple", "orange", "banana"});
   test("apple::orange::banana", "::", {"apple", "orange", "banana"});
   test("appleorangebanana", "", {"apple", "orange", "banana"});
   test("apple,,banana", ",", {"apple", "", "banana"});
   test("apple", ",", {"apple"});
   test("", ",", {""});
   test("", "", {""});
   test("", "", {});
   test("", ";;", {""});
}

TEST(StringUtils, Round)
{
   EXPECT_EQ(ROOT::Round(0.000000014, 0.000000024), "(10#pm20)*1e-9");
   EXPECT_EQ(ROOT::Round(0.000000014, 0.000000018), "(14#pm18)*1e-9");
   EXPECT_EQ(ROOT::Round(110., 0.24), "110.0#pm0.2");
   EXPECT_EQ(ROOT::Round(120., 0.24, 2), "120.00#pm0.24");
   EXPECT_EQ(ROOT::Round(130., 0.94), "130.0#pm0.9");
   EXPECT_EQ(ROOT::Round(-140., 0.95), "-140.0#pm1.0");
   EXPECT_EQ(ROOT::Round(150., 0.114), "150.00#pm0.11");
   EXPECT_EQ(ROOT::Round(-160., 0.194), "-160.00#pm0.19");
   EXPECT_EQ(ROOT::Round(170., 0.195), "170.0#pm0.2");
   EXPECT_EQ(ROOT::Round(-180., 0.94), "-180.0#pm0.9");
   EXPECT_EQ(ROOT::Round(190., 0.95), "190.0#pm1.0");
   EXPECT_EQ(ROOT::Round(-190., 0.95, 0), "-190#pm1");
   EXPECT_EQ(ROOT::Round(200., 0.95, 0), "200#pm1");
   EXPECT_EQ(ROOT::Round(-210., 2.4), "-210#pm2");
   EXPECT_EQ(ROOT::Round(220., 9.4), "220#pm9");
   EXPECT_EQ(ROOT::Round(-0.001, 9.5), "-0#pm10");
   EXPECT_EQ(ROOT::Round(230., 11.4), "230#pm11");
   EXPECT_EQ(ROOT::Round(24., 19.4), "24#pm19");
   EXPECT_EQ(ROOT::Round(-25., 19.5), "-30#pm20");
   EXPECT_EQ(ROOT::Round(-25., 21, 9), "-25#pm21");
   EXPECT_EQ(ROOT::Round(280., 94), "280#pm90");
   EXPECT_EQ(ROOT::Round(-190., 95), "-190#pm100");
   EXPECT_EQ(ROOT::Round(1., 101.4), "0#pm100");
   EXPECT_EQ(ROOT::Round(-1., 109.4), "-0#pm110");
   EXPECT_EQ(ROOT::Round(300., 119.4), "300#pm120");
   EXPECT_EQ(ROOT::Round(-31., 119.5), "-30#pm120");
   EXPECT_EQ(ROOT::Round(320., 194), "320#pm190");
   EXPECT_EQ(ROOT::Round(-3030., 195), "-3000#pm200");
   EXPECT_EQ(ROOT::Round(1400., 201), "1400#pm200");
   EXPECT_EQ(ROOT::Round(-1200., 2000), "(-1#pm2)*1e3");
   EXPECT_EQ(ROOT::Round(101., 2000, 2), "(0.1#pm2.0)*1e3");
   EXPECT_EQ(ROOT::Round(-5056., 194, 9), "-5060#pm190");
   EXPECT_EQ(ROOT::Round(-30000., 2000000000., 2), "(-0.0#pm2.0)*1e9");
   EXPECT_EQ(ROOT::Round(-30000., 1000000000., 99), "(-0.0#pm1.0)*1e9");
   EXPECT_EQ(ROOT::Round(-30000., 1000000000., 0), "(-0#pm1)*1e9");
   EXPECT_EQ(ROOT::Round(110., 0.24, 1, "+-"), "110.0+-0.2");
}
