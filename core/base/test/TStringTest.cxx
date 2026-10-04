#include "gtest/gtest.h"
#include "ROOT/TestSupport.hxx"
#include "TString.h"
#include "TObjArray.h"
#include "TObjString.h"

TEST(TString, Basics)
{
   TString *s = nullptr;
   ROOT_EXPECT_ERROR(s = new TString("Test", -5), "TString::TString", "Negative length!");
   delete s;
   TString p("Test", 1);
   EXPECT_STREQ("T", p.Data());
   TString a = "test";
   ROOT_EXPECT_ERROR(a.Append("s", -5), "TString::Replace", "Negative number of replacement characters!");
   EXPECT_STREQ("test", a.Data());
}

TEST(TString, Contains)
{
   TString s("HelloWorld");
   EXPECT_TRUE(s.Contains("World"));
   EXPECT_TRUE(s.Contains("hello", TString::kIgnoreCase));
   EXPECT_FALSE(s.Contains("hello", TString::kExact));
}

TEST(TString, ReplaceAll)
{
   TString s("The quick brown fox");
   s.ReplaceAll("quick", "slow");
   EXPECT_STREQ("The slow brown fox", s.Data());

   s.ReplaceAll("fox", "dog");
   EXPECT_STREQ("The slow brown dog", s.Data());
}

TEST(TString, CaseConversion)
{
   TString s("ROOT Framework");
   s.ToUpper();
   EXPECT_STREQ("ROOT FRAMEWORK", s.Data());
   s.ToLower();
   EXPECT_STREQ("root framework", s.Data());
}

TEST(TString, Formatting)
{
   TString s;
   s.Form("Number %d and string %s", 42, "test");
   EXPECT_STREQ("Number 42 and string test", s.Data());
}

TEST(TString, Substring)
{
   TString s("abcdef");
   TString sub = s(1, 3);
   EXPECT_STREQ("bcd", sub.Data());
}

TEST(TString, Tokenize)
{
   TString s("apple,banana,orange");
   std::unique_ptr<TObjArray> tokens(s.Tokenize(","));
   ASSERT_EQ(3, tokens->GetEntries());
   EXPECT_STREQ("apple", ((TObjString*)tokens->At(0))->GetString().Data());
   EXPECT_STREQ("banana", ((TObjString*)tokens->At(1))->GetString().Data());
   EXPECT_STREQ("orange", ((TObjString*)tokens->At(2))->GetString().Data());
}

