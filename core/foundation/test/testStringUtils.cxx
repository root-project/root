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
