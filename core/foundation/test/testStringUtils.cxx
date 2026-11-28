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

TEST(StringUtils, JoinStringViews)
{
   const std::string source = "appleorangebanana";
   const std::vector<std::string_view> strings = {std::string_view(source).substr(0, 5),
                                                  std::string_view(source).substr(5, 6),
                                                  std::string_view(source).substr(11)};
   EXPECT_EQ(ROOT::Join("::", strings), "apple::orange::banana");
   EXPECT_EQ(ROOT::Join("", strings.begin(), strings.end()), source);
}

TEST(StringUtils, JoinCStringArray)
{
   const char *strings[] = {"apple", "", "banana"};
   EXPECT_EQ(ROOT::Join(",", strings), "apple,,banana");
}

TEST(StringUtils, JoinStringViewSeparator)
{
   const std::string separators = "--::--";
   const std::vector<std::string> strings = {"apple", "orange"};
   EXPECT_EQ(ROOT::Join(std::string_view(separators).substr(2, 2), strings), "apple::orange");
   EXPECT_EQ(ROOT::Join(std::string(", "), strings), "apple, orange");
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

TEST(StringUtils, SplitAt)
{
   const auto p = [](auto a, auto b) { return std::make_pair<std::string_view, std::string_view>(a, b); };

   EXPECT_EQ(ROOT::SplitAt("foo, bar", ','), p("foo", " bar"));
   EXPECT_EQ(ROOT::SplitAt("foo, bar,", ','), p("foo", " bar,"));
   EXPECT_EQ(ROOT::SplitAt(", bar,", ','), p("", " bar,"));
   EXPECT_EQ(ROOT::SplitAt("foo,bar", ' '), p("foo,bar", ""));
   EXPECT_EQ(ROOT::SplitAt("", ' '), p("", ""));
   EXPECT_EQ(ROOT::SplitAt(",,,,", ','), p("", ",,,"));
}

TEST(StringUtils, WrapString)
{
   const auto input = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
                      "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
                      "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in "
                      "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
                      "non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

   EXPECT_EQ(ROOT::WrapString(input, 60),
             R"(Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation
ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
occaecat cupidatat non proident, sunt in culpa qui officia
deserunt mollit anim id est laborum.)");
   EXPECT_EQ(
      ROOT::WrapString(input, 120),
      R"(Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis
aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.)");
   EXPECT_EQ(ROOT::WrapString(input, 60, 10),
             R"(Lorem ipsum dolor sit amet, consectetur adipiscing
          elit, sed do eiusmod tempor incididunt ut labore
          et dolore magna aliqua. Ut enim ad minim veniam,
          quis nostrud exercitation ullamco laboris nisi ut
          aliquip ex ea commodo consequat. Duis aute irure
          dolor in reprehenderit in voluptate velit esse
          cillum dolore eu fugiat nulla pariatur. Excepteur
          sint occaecat cupidatat non proident, sunt in
          culpa qui officia deserunt mollit anim id est
          laborum.)");
   EXPECT_EQ(ROOT::WrapString(input, 60, /*indent=*/10, /*firstLineMaxLen=*/20),
             R"(Lorem ipsum dolor
          sit amet, consectetur adipiscing elit, sed do
          eiusmod tempor incididunt ut labore et dolore
          magna aliqua. Ut enim ad minim veniam, quis
          nostrud exercitation ullamco laboris nisi ut
          aliquip ex ea commodo consequat. Duis aute irure
          dolor in reprehenderit in voluptate velit esse
          cillum dolore eu fugiat nulla pariatur. Excepteur
          sint occaecat cupidatat non proident, sunt in
          culpa qui officia deserunt mollit anim id est
          laborum.)");
   EXPECT_EQ(ROOT::WrapString("A shorter string, just to get the point across...", 1),
             R"(A
shorter
string,
just
to
get
the
point
across...)");
   const auto shortStr = "Another short string, not long enough to be wrapped in this case";
   EXPECT_EQ(ROOT::WrapString(shortStr, 200), shortStr);
}

TEST(StringUtils, WrapStringNewlines)
{
   const auto inputWithNewlines =
      "\n\n" // should be preserved
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut"
      "\n" // should not be preserved
      "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco"
      "\n"
      "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in"
      "\n\n\n\n"
      "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur\nsint occaecat cupidatat "
      "non proident, sunt in culpa qui officia"
      "\n\n"
      "deserunt mollit anim id est laborum."
      "\n\n";
   EXPECT_EQ(ROOT::WrapString(inputWithNewlines, 60),
             R"(

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation
ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in



voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in
culpa qui officia

deserunt mollit anim id est laborum.

)");

   const auto inputWithNewlines2 =
      R"(
word0 word1

word2 word3 

word4 word5 word6

 word7 word8 word9 wor10 wor11 wor12 wor13 wor14 wor15 wor16 wor17 wor18 wor19 wor20



)";
   EXPECT_EQ(ROOT::WrapString(inputWithNewlines2, 80),
             R"(word0 word1

word2 word3 

word4 word5 word6

 word7 word8 word9 wor10 wor11 wor12 wor13 wor14 wor15 wor16 wor17 wor18 wor19
wor20



)");
}
