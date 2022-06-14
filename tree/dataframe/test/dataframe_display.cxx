/****** Run RDataFrame tests both with and without IMT enabled *******/
#include <gtest/gtest.h>
#include <ROOT/RDataFrame.hxx>
#include <TTree.h>

#include <utility> // std::pair

using namespace ROOT;
using namespace ROOT::RDF;
using namespace ROOT::VecOps;

static const std::string DisplayPrintDefaultRows("+-----+----+----+-----------+\n"
                                                 "| Row | b1 | b2 | b3        | \n"
                                                 "+-----+----+----+-----------+\n"
                                                 "| 0   | 0  | 1  | 2.0000000 | \n"
                                                 "|     |    | 2  |           | \n"
                                                 "|     |    | 3  |           | \n"
                                                 "+-----+----+----+-----------+\n"
                                                 "| 1   | 0  | 1  | 2.0000000 | \n"
                                                 "|     |    | 2  |           | \n"
                                                 "|     |    | 3  |           | \n"
                                                 "+-----+----+----+-----------+\n"
                                                 "| 2   | 0  | 1  | 2.0000000 | \n"
                                                 "|     |    | 2  |           | \n"
                                                 "|     |    | 3  |           | \n"
                                                 "+-----+----+----+-----------+\n"
                                                 "| 3   | 0  | 1  | 2.0000000 | \n"
                                                 "|     |    | 2  |           | \n"
                                                 "|     |    | 3  |           | \n"
                                                 "+-----+----+----+-----------+\n"
                                                 "| 4   | 0  | 1  | 2.0000000 | \n"
                                                 "|     |    | 2  |           | \n"
                                                 "|     |    | 3  |           | \n"
                                                 "+-----+----+----+-----------+\n"
);

static const std::string DisplayAsStringDefaultRows("+-----+----+----+-----------+\n"
                                                    "| Row | b1 | b2 | b3        | \n"
                                                    "+-----+----+----+-----------+\n"
                                                    "| 0   | 0  | 1  | 2.0000000 | \n"
                                                    "|     |    | 2  |           | \n"
                                                    "|     |    | 3  |           | \n"
                                                    "+-----+----+----+-----------+\n"
                                                    "| 1   | 0  | 1  | 2.0000000 | \n"
                                                    "|     |    | 2  |           | \n"
                                                    "|     |    | 3  |           | \n"
                                                    "+-----+----+----+-----------+\n"
                                                    "| 2   | 0  | 1  | 2.0000000 | \n"
                                                    "|     |    | 2  |           | \n"
                                                    "|     |    | 3  |           | \n"
                                                    "+-----+----+----+-----------+\n"
                                                    "| 3   | 0  | 1  | 2.0000000 | \n"
                                                    "|     |    | 2  |           | \n"
                                                    "|     |    | 3  |           | \n"
                                                    "+-----+----+----+-----------+\n"
                                                    "| 4   | 0  | 1  | 2.0000000 | \n"
                                                    "|     |    | 2  |           | \n"
                                                    "|     |    | 3  |           | \n"
                                                    "|     |    |    |           | \n"
                                                    "+-----+----+----+-----------+\n"
);

TEST(RDFDisplayTests, DisplayNoJitDefaultRows)
{
   RDataFrame rd1(10);
   auto dd = rd1.Define("b1", []() { return 0; })
                .Define("b2",
                        []() {
                           return std::vector<int>({1, 2, 3});
                        })
                .Define("b3", []() { return 2.; })
                .Display<int, std::vector<int>, double>({"b1", "b2", "b3"});

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(), DisplayPrintDefaultRows);

   // Testing the string returned
   EXPECT_EQ(dd->AsString(), DisplayAsStringDefaultRows);
}

TEST(RDFDisplayTests, DisplayJitDefaultRows)
{
   RDataFrame rd1(10);
   auto dd = rd1.Define("b1", []() { return 0; })
                .Define("b2",
                        []() {
                           return std::vector<int>({1, 2, 3});
                        })
                .Define("b3", []() { return 2.; })
                .Display({"b1", "b2", "b3"});

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(), DisplayPrintDefaultRows);

   // Testing the string returned
   EXPECT_EQ(dd->AsString(), DisplayAsStringDefaultRows);
}

TEST(RDFDisplayTests, DisplayRegexDefaultRows)
{
   RDataFrame rd1(10);
   auto dd = rd1.Define("b1", []() { return 0; })
                .Define("b2",
                        []() {
                           return std::vector<int>({1, 2, 3});
                        })
                .Define("b3", []() { return 2.; })
                .Display("");

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(), DisplayPrintDefaultRows);

   // Testing the string returned
   EXPECT_EQ(dd->AsString(), DisplayAsStringDefaultRows);
}

static const std::string DisplayPrintTwoRows("+-----+----+----+-----------+\n"
                                             "| Row | b1 | b2 | b3        | \n"
                                             "+-----+----+----+-----------+\n"
                                             "| 0   | 0  | 1  | 2.0000000 | \n"
                                             "|     |    | 2  |           | \n"
                                             "|     |    | 3  |           | \n"
                                             "+-----+----+----+-----------+\n"
                                             "| 1   | 0  | 1  | 2.0000000 | \n"
                                             "|     |    | 2  |           | \n"
                                             "|     |    | 3  |           | \n"
                                             "+-----+----+----+-----------+\n");

static const std::string DisplayAsStringTwoRows("+-----+----+----+-----------+\n"
                                                "| Row | b1 | b2 | b3        | \n"
                                                "+-----+----+----+-----------+\n"
                                                "| 0   | 0  | 1  | 2.0000000 | \n"
                                                "|     |    | 2  |           | \n"
                                                "|     |    | 3  |           | \n"
                                                "+-----+----+----+-----------+\n"
                                                "| 1   | 0  | 1  | 2.0000000 | \n"
                                                "|     |    | 2  |           | \n"
                                                "|     |    | 3  |           | \n"
                                                "|     |    |    |           | \n"
                                                "+-----+----+----+-----------+\n");

TEST(RDFDisplayTests, DisplayJitTwoRows)
{
   RDataFrame rd1(10);
   auto dd = rd1.Define("b1", []() { return 0; })
                .Define("b2",
                        []() {
                           return std::vector<int>({1, 2, 3});
                        })
                .Define("b3", []() { return 2.; })
                .Display({"b1", "b2", "b3"}, 2);

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(), DisplayPrintTwoRows);

   // Testing the string returned
   EXPECT_EQ(dd->AsString(), DisplayAsStringTwoRows);
}

static const std::string DisplayAsStringOneColumn("+-----+----+\n"
                                                  "| Row | b1 | \n"
                                                  "+-----+----+\n"
                                                  "| 0   | 0  | \n"
                                                  "+-----+----+\n"
                                                  "| 1   | 0  | \n"
                                                  "+-----+----+\n"
                                                  "| 2   | 0  | \n"
                                                  "+-----+----+\n"
                                                  "| 3   | 0  | \n"
                                                  "+-----+----+\n"
                                                  "| 4   | 0  | \n"
                                                  "|     |    | \n"
                                                  "+-----+----+\n");
static const std::string DisplayAsStringTwoColumns("+-----+----+----+\n"
                                                   "| Row | b1 | b2 | \n"
                                                   "+-----+----+----+\n"
                                                   "| 0   | 0  | 1  | \n"
                                                   "|     |    | 2  | \n"
                                                   "|     |    | 3  | \n"
                                                   "+-----+----+----+\n"
                                                   "| 1   | 0  | 1  | \n"
                                                   "|     |    | 2  | \n"
                                                   "|     |    | 3  | \n"
                                                   "+-----+----+----+\n"
                                                   "| 2   | 0  | 1  | \n"
                                                   "|     |    | 2  | \n"
                                                   "|     |    | 3  | \n"
                                                   "+-----+----+----+\n"
                                                   "| 3   | 0  | 1  | \n"
                                                   "|     |    | 2  | \n"
                                                   "|     |    | 3  | \n"
                                                   "+-----+----+----+\n"
                                                   "| 4   | 0  | 1  | \n"
                                                   "|     |    | 2  | \n"
                                                   "|     |    | 3  | \n"
                                                   "|     |    |    | \n"
                                                   "+-----+----+----+\n");

TEST(RDFDisplayTests, DisplayAmbiguity)
{
   // This test verifies that the correct method is called and there is no ambiguity between the JIT call to Display
   // using a column list as a parameter and the JIT call to Display using the Regexp.
   RDataFrame rd1(10);
   auto dd = rd1.Define("b1", []() { return 0; }).Define("b2", []() { return std::vector<int>({1, 2, 3}); });

   auto display_1 = dd.Display({"b1"});
   auto display_2 = dd.Display({"b1", "b2"});

   EXPECT_EQ(display_1->AsString(), DisplayAsStringOneColumn);
   EXPECT_EQ(display_2->AsString(), DisplayAsStringTwoColumns);
}

static const std::string DisplayAsStringString("+-----+-------+\n| Row | b1    | \n+-----+-------+\n| 0   | \"foo\" | \n+-----+-------+\n| 1   | \"foo\" | \n|     |       | \n+-----+-------+\n");

TEST(RDFDisplayTests, DisplayPrintString)
{
   RDataFrame rd1(2);
   auto dd = rd1.Define("b1", []() { return std::string("foo"); })
                .Display({"b1"});

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   // Testing the string returned
   EXPECT_EQ(dd->AsString(), DisplayAsStringString);
}

TEST(RDFDisplayTests, CharArray)
{
   {
      TFile f("chararray.root", "recreate");
      TTree t("t", "t");
      char str[4] = "asd";
      t.Branch("str", str, "str[4]/C");
      t.Fill();
      char otherstr[4] = "bar";
      std::copy(otherstr, otherstr + 4, str);
      t.Fill();
      f.Write();
   }

   const auto str = ROOT::RDataFrame("t", "chararray.root").Display()->AsString();
   EXPECT_EQ(str, "+-----+-----+\n| Row | str | \n+-----+-----+\n| 0   | asd | \n+-----+-----+\n| 1   | bar | \n|     |     | \n+-----+-----+\n");
}

TEST(RDFDisplayTests, BoolArray)
{
   auto r = ROOT::RDataFrame(3)
      .Define("v", [] { return ROOT::RVec<bool>{true,false}; })
      .Display<ROOT::RVec<bool>>({"v"});
   const auto expected = "+-----+-------+\n"
                         "| Row | v     | \n"
                         "+-----+-------+\n"
                         "| 0   | true  | \n"
                         "|     | false | \n"
                         "+-----+-------+\n"
                         "| 1   | true  | \n"
                         "|     | false | \n"
                         "|     | true  | \n"
                         "|     | false | \n"
                         "+-----+-------+\n"
                         "| 2   | true  | \n"
                         "|     | false | \n"
                         "|     | true  | \n"
                         "|     | false | \n"
                         "|     | true  | \n"
                         "|     | false | \n"
                         "|     |       | \n"
                         "+-----+-------+\n";
   EXPECT_EQ(r->AsString(), expected);
}

TEST(RDFDisplayTests, UniquePtr)
{
   auto r = ROOT::RDataFrame(1)
               .Define("uptr", []() -> std::unique_ptr<int> { return nullptr; })
               .Display<std::unique_ptr<int>>({"uptr"});
   const auto expected = "+-----+----------------------------+\n"
                         "| Row | uptr                       | \n"
                         "+-----+----------------------------+\n"
                         "| 0   | std::unique_ptr -> nullptr | \n"
                         "|     |                            | \n"
                         "+-----+----------------------------+\n";
   EXPECT_EQ(r->AsString(), expected);
}


// GitHub issue #6371
TEST(RDFDisplayTests, SubBranch)
{
   auto p = std::make_pair(42, 84);
   TTree t("t", "t");
   t.Branch("p", &p, "a/I:b/I");
   t.Fill();
   ROOT::RDataFrame df(t);
   const auto res = df.Display()->AsString();
   const auto expected = "+-----+-----+-----+\n| Row | p.a | p.b | \n+-----+-----+-----+\n| 0   | 42  | 84  | \n|     |     |     | \n+-----+-----+-----+\n";
   EXPECT_EQ(res, expected);
}

// https://github.com/root-project/root/issues/8450
TEST(RDFDisplayTests, Friends)
{
  TTree main("main", "main");
  main.Fill();
  TTree fr("friend", "friend");
  int x = 0;
  fr.Branch("x", &x);
  fr.Fill();
  main.AddFriend(&fr);

  const auto res = ROOT::RDataFrame(main).Display()->AsString();
  const auto expected = "+-----+----------+\n| Row | friend.x | \n+-----+----------+\n| 0   | 0        | \n|     |          | \n+-----+----------+\n";
  EXPECT_EQ(res, expected);
}

static const std::string DisplayPrintVectors("+-----+----+----+----+----+-----+-----+-----+------+\n"
                                             "| Row | S0 | S1 | S3 | S9 | S10 | S11 | S20 | S20_ | \n"
                                             "+-----+----+----+----+----+-----+-----+-----+------+\n"
                                             "| 0   |    | 0  | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    |    | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    |    |     | ... | ... | ...  | \n"
                                             "+-----+----+----+----+----+-----+-----+-----+------+\n"
                                             "| 1   |    | 0  | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    |    | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    |    |     | ... | ... | ...  | \n"
                                             "+-----+----+----+----+----+-----+-----+-----+------+\n"
                                             "| 2   |    | 0  | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    |    | 0   | 0   | 0   | 0    | \n"
                                             "|     |    |    |    |    |     | ... | ... | ...  | \n"
                                             "+-----+----+----+----+----+-----+-----+-----+------+\n");

static const std::string DisplayAsStringVectors("+-----+----+----+----+----+-----+-----+-----+------+\n"
                                                "| Row | S0 | S1 | S3 | S9 | S10 | S11 | S20 | S20_ | \n"
                                                "+-----+----+----+----+----+-----+-----+-----+------+\n"
                                                "| 0   |    | 0  | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    |     | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "+-----+----+----+----+----+-----+-----+-----+------+\n"
                                                "| 1   |    | 0  | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    |     | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "+-----+----+----+----+----+-----+-----+-----+------+\n"
                                                "| 2   |    | 0  | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    | 0  | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    | 0  | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    | 0   | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    |     | 0   | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     | 0   | 0    | \n"
                                                "|     |    |    |    |    |     |     |     |      | \n"
                                                "+-----+----+----+----+----+-----+-----+-----+------+\n");

TEST(RDFDisplayTests, Vectors)
{
   std::vector<int> v0(0);
   std::vector<int> v1(1);
   std::vector<int> v3(3);
   std::vector<int> v9(9);
   std::vector<int> v10(10);
   std::vector<int> v11(11);
   std::vector<int> v20(20);

   RDataFrame vc(3);

   auto dd = vc.Define("S0", [&v0] { return v0; })
                .Define("S1", [&v1] { return v1; })
                .Define("S3", [&v3] { return v3; })
                .Define("S9", [&v9] { return v9; })
                .Define("S10", [&v10] { return v10; })
                .Define("S11", [&v11] { return v11; })
                .Define("S20", [&v20] { return v20; })
                .Define("S20_", [&v20] { return v20; })
                .Display<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>,
                         std::vector<int>, std::vector<int>, std::vector<int>>(
                   {"S0", "S1", "S3", "S9", "S10", "S11", "S20", "S20_"});

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(), DisplayPrintVectors);

   // Testing the string returned
   EXPECT_EQ(dd->AsString(), DisplayAsStringVectors);
}

TEST(RDFDisplayTests, CustomMaxWidth)
{

   std::vector<int> v3(3);
   ROOT::RDataFrame vc(3);
   auto dd = vc.Define("S3", [&v3] { return v3; }).Display<std::vector<int>>({"S3"}, 1, 2);

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(), "+-----+-----+\n| Row | S3  | \n+-----+-----+\n| 0   | 0   | \n|     | 0   | \n|     | ... | \n+-----+-----+\n");
}

TEST(RDFDisplayTests, PrintWideTables1)
{
  std::vector<std::string> v3(3);
  v3[0] = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890";
  v3[1] = v3[0];
  v3[2] = v3[0];
  std::vector<int> v0(10);
  ROOT::RDataFrame vc(3);
  auto dd = vc.Define("S3", [&v3] { return v3; })
              .Display<std::vector<std::string>>({"S3"});

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(),
   "+-----+--------------------------------------------------------------------------------------------------------+\n"
   "| Row | S3                                                                                                     | \n"
   "+-----+--------------------------------------------------------------------------------------------------------+\n"
   "| 0   | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "|     | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "|     | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "+-----+--------------------------------------------------------------------------------------------------------+\n"
   "| 1   | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "|     | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "|     | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "+-----+--------------------------------------------------------------------------------------------------------+\n"
   "| 2   | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "|     | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "|     | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | \n"
   "+-----+--------------------------------------------------------------------------------------------------------+\n");
}

TEST(RDFDisplayTests, PrintWideTables2)
{
  std::vector<std::string> v3(3);
  v3[0] = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890";
  v3[1] = v3[0];
  v3[2] = v3[0];
  std::vector<int> v10(10);
  ROOT::RDataFrame vc(1);
  auto dd = vc.Define("S10", [&v10] { return v10; })
              .Define("S3", [&v3] { return v3; })
              .Display<std::vector<int>, std::vector<std::string>>({"S10", "S3"});

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(), "+-----+-----+-----+\n"
                            "| Row | S10 | ... | \n"
                            "+-----+-----+-----+\n"
                            "| 0   | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "|     | 0   | ... | \n"
                            "+-----+-----+-----+\n");
}


TEST(RDFDisplayTests, PrintWideTables3)
{
  std::vector<std::string> v3(3);
  v3[0] = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890";
  v3[1] = v3[0];
  v3[2] = v3[0];
  std::vector<int> v10(10);
  ROOT::RDataFrame vc(1);
  auto dd = vc.Define("S10", [&v10] { return v10; })
              .Define("S3", [&v3] { return v3; })
              .Display<std::vector<std::string>, std::vector<int>>({"S3", "S10"});

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   dd->Print();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);

   EXPECT_EQ(strCout.str(),
   "+-----+--------------------------------------------------------------------------------------------------------+-----+\n"
   "| Row | S3                                                                                                     | ... | \n"
   "+-----+--------------------------------------------------------------------------------------------------------+-----+\n"
   "| 0   | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | ... | \n"
   "|     | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | ... | \n"
   "|     | \"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod 12345678901234567890\" | ... | \n"
   "+-----+--------------------------------------------------------------------------------------------------------+-----+\n");
}

