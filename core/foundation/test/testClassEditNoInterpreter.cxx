// Tests for TClassEdit that must run without the interpreter: as soon as
// TCling is initialized (e.g. by using gInterpreter), it registers its lookup
// helper with TClassEdit, and the name normalization falls back to clang's
// desugaring, which can hide bugs in TClassEdit's own string manipulation.
// Therefore, nothing in this file must touch gInterpreter, gROOT or TClass.

#include "TClassEdit.h"

#include "gtest/gtest.h"

class TClassEditNoInterpreter : public ::testing::Test {
protected:
   void SetUp() override
   {
      // Check the premise of this file instead of relying on it: without a
      // lookup helper, ResolveTypedef() returns its input unchanged, while
      // TCling's helper resolves the typedef.
      ASSERT_EQ("Int_t", TClassEdit::ResolveTypedef("Int_t"))
         << "TClassEdit has an interpreter lookup helper, so this test can't see bugs in TClassEdit itself";
   }
};

// Part of https://github.com/root-project/root/issues/19940
TEST_F(TClassEditNoInterpreter, NormalizeConstString)
{
   std::string n;

   TClassEdit::GetNormalizedName(n, "const basic_string<char,char_traits<char>,allocator<char> >");
   EXPECT_EQ("const string", n);

   TClassEdit::GetNormalizedName(n, "const std::basic_string<char, std::char_traits<char>, std::allocator<char> >");
   EXPECT_EQ("const string", n);

   TClassEdit::GetNormalizedName(n, "const std::__cxx11::basic_string<char>");
   EXPECT_EQ("const string", n);

   TClassEdit::GetNormalizedName(n, "basic_string<char,char_traits<char>,allocator<char> > const");
   EXPECT_EQ("const string", n);

   TClassEdit::GetNormalizedName(n, "vector<const basic_string<char,char_traits<char>,allocator<char> > >");
   EXPECT_EQ("vector<const string>", n);

   TClassEdit::GetNormalizedName(n, "std::map<const std::basic_string<char>, int>");
   EXPECT_EQ("map<const string,int>", n);
}
