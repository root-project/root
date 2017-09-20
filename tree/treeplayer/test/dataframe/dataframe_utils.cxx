#include "ROOT/TDataFrame.hxx"
#include "ROOT/TDFUtils.hxx"
#include "TTree.h"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

// Thanks clang-format...
TEST(TDataFrameUtils, DeduceAllPODsFromTmpColumns)
{
   TDataFrame tdf(1);
   auto d = tdf.Define("char_tmp",
                       []() {
                          char v(0);
                          return v;
                       })
               .Define("uchar_tmp",
                       []() {
                          unsigned char v(0);
                          return v;
                       })
               .Define("int_tmp",
                       []() {
                          int v(0);
                          return v;
                       })
               .Define("uint_tmp",
                       []() {
                          unsigned int v(0);
                          return v;
                       })
               .Define("short_tmp",
                       []() {
                          short v(0);
                          return v;
                       })
               .Define("ushort_tmp",
                       []() {
                          unsigned short v(0);
                          return v;
                       })
               .Define("double_tmp",
                       []() {
                          double v(0);
                          return v;
                       })
               .Define("float_tmp",
                       []() {
                          float v(0);
                          return v;
                       })
               .Define("Long64_t_tmp",
                       []() {
                          Long64_t v(0);
                          return v;
                       })
               .Define("ULong64_t_tmp",
                       []() {
                          ULong64_t v(0);
                          return v;
                       })
               .Define("bool_tmp", []() {
                  bool v(0);
                  return v;
               });
   auto c = tdf.Snapshot<char, unsigned char, int, unsigned int, short, unsigned short, double, float, Long64_t,
                         ULong64_t, bool>("t", "dataframe_interfaceAndUtils_1.root",
                                          {"char_tmp", "uchar_tmp", "int_tmp", "uint_tmp", "short_tmp", "ushort_tmp",
                                           "double_tmp", "float_tmp", "Long64_t_tmp", "ULong64_t_tmp", "bool_tmp"});
}

TEST(TDataFrameUtils, DeduceAllPODsFromColumns)
{
   char c;
   unsigned char uc;
   int i;
   unsigned int ui;
   short s;
   unsigned short us;
   double d;
   float f;
   Long64_t l;
   ULong64_t ul;
   bool b;
   int a[2];

   TTree t("t", "t");
   t.Branch("char", &c);
   t.Branch("uchar", &uc);
   t.Branch("i", &i);
   t.Branch("uint", &ui);
   t.Branch("short", &s);
   t.Branch("ushort", &us);
   t.Branch("double", &d);
   t.Branch("float", &f);
   t.Branch("Long64_t", &l);
   t.Branch("ULong64_t", &ul);
   t.Branch("bool", &b);
   t.Branch("arrint", &a, "a[2]/I");
   t.Branch("vararrint", &a, "a[i]/I");

   std::map<const char *, const char *> nameTypes = {{"char", "char"},
                                                     {"uchar", "unsigned char"},
                                                     {"i", "int"},
                                                     {"uint", "unsigned int"},
                                                     {"short", "short"},
                                                     {"ushort", "unsigned short"},
                                                     {"double", "double"},
                                                     {"float", "float"},
                                                     {"Long64_t", "Long64_t"},
                                                     {"ULong64_t", "ULong64_t"},
                                                     {"bool", "bool"},
                                                     {"arrint", "std::array_view<int>"},
                                                     {"vararrint", "std::array_view<int>"}};

   for (auto &nameType : nameTypes) {
      auto typeName = ROOT::Internal::TDF::ColumnName2ColumnTypeName(nameType.first, &t, nullptr);
      EXPECT_STREQ(nameType.second, typeName.c_str());
   }
}

TEST(TDataFrameUtils, ToConstCharPtr)
{
   const char *s_content("mystring");
   std::string s("mystring");
   EXPECT_STREQ(s_content, ROOT::Internal::TDF::ToConstCharPtr(s_content));
   EXPECT_STREQ(s_content, ROOT::Internal::TDF::ToConstCharPtr(s));
}


TEST(TDataFrameUtils, CheckNonExistingCustomColumnNullTree)
{
   // CheckCustomColumn(std::string_view definedCol, TTree *treePtr, const ColumnNames_t &customCols,
   //                   const ColumnNames_t &dataSourceColumns)
   ROOT::Internal::TDF::CheckCustomColumn("Bla", nullptr, {"a", "b"}, {});
}

TEST(TDataFrameUtils, CheckExistingCustomColumnNullTree)
{
   int ret = 1;
   try {
      ROOT::Internal::TDF::CheckCustomColumn("a", nullptr, {"a", "b"}, {});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(TDataFrameUtils, CheckExistingCustomColumn)
{
   int i;
   TTree t("t", "t");
   t.Branch("a", &i);

   int ret = 1;
   try {
      ROOT::Internal::TDF::CheckCustomColumn("a", &t, {"b"}, {});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(TDataFrameUtils, CheckExistingCustomColumnDataSource)
{
   int i;
   TTree t("t", "t");
   t.Branch("a", &i);

   int ret = 1;
   try {
      ROOT::Internal::TDF::CheckCustomColumn("c", &t, {"b"}, {"c"});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(TDataFrameUtils, CheckSnapshot)
{
   int ret = 1;
   try {
      ROOT::Internal::TDF::CheckSnapshot(5, 4);
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(TDataFrameUtils, SelectColumnsNNamesDiffersRequiredNames)
{
   int ret = 1;
   try {
      ROOT::Internal::TDF::SelectColumns(3, {"a", "b"}, {});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(TDataFrameUtils, SelectColumnsTooFewRequiredNames)
{
   int ret = 1;
   try {
      ROOT::Internal::TDF::SelectColumns(3, {}, {"bla"});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(TDataFrameUtils, SelectColumnsCheckNames)
{
   ROOT::Internal::TDF::ColumnNames_t cols{"a", "b", "c"};
   auto ncols = ROOT::Internal::TDF::SelectColumns(2, {}, cols);
   EXPECT_STREQ("a", ncols[0].c_str());
   EXPECT_STREQ("b", ncols[1].c_str());
}

TEST(TDataFrameUtils, FindUnknownColumns)
{
   int i;
   TTree t("t", "t");
   t.Branch("a", &i);

   auto ncols = ROOT::Internal::TDF::FindUnknownColumns({"a", "b", "c", "d"}, &t, {"b"}, {});
   EXPECT_STREQ("c", ncols[0].c_str());
   EXPECT_STREQ("d", ncols[1].c_str());
}

TEST(TDataFrameUtils, FindUnknownColumnsWithDataSource)
{
   int i;
   TTree t("t", "t");
   t.Branch("a", &i);

   auto ncols = ROOT::Internal::TDF::FindUnknownColumns({"a", "b", "c", "d"}, &t, {"b"}, {"c"});
   EXPECT_EQ(ncols.size(), 1u);
   EXPECT_STREQ("d", ncols[0].c_str());
}
