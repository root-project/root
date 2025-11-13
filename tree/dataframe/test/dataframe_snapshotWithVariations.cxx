#include <ROOT/RDataFrame.hxx>
#include "TwoFloats_Snapshot.h"

#include <TBranch.h>
#include <TDataType.h>
#include <TFile.h>
#include <TInterpreter.h>
#include <TLeaf.h>
#include <TTree.h>
#include <TSystem.h>

#include <gtest/gtest.h>

#include <bitset>
#include <filesystem>
#include <numeric>

constexpr bool verbose = false;

using ROOT::RDF::Experimental::VariationsFor;

#define RemoveFileRAII(...)                      \
   struct RFR {                                  \
      std::vector<std::string> filenames;        \
      ~RFR()                                     \
      {                                          \
         if (!HasFailure()) {                    \
            for (auto const &file : filenames) { \
               std::remove(file.c_str());        \
            }                                    \
         }                                       \
      }                                          \
   } rfr{{__VA_ARGS__}};

void printTree(TTree &tree)
{
   tree.Print();
   tree.SetScanField(50);
   tree.Scan("*", "", "colsize=15", 50);
}

template <typename X_t, typename Y_t, typename F>
void checkOutput(TTree &tree, std::vector<std::string> const &systematics, F &&activeCuts)
{
   X_t x;
   Y_t y;

   for (auto const &sysName : systematics) {
      ASSERT_EQ(TTree::kMatch, tree.SetBranchAddress(("x" + sysName).c_str(), &x)) << ("x" + sysName);
      ASSERT_EQ(TTree::kMatch, tree.SetBranchAddress(("y" + sysName).c_str(), &y)) << ("y" + sysName);

      for (unsigned int i = 0; i < tree.GetEntries(); ++i) {
         ASSERT_GT(tree.GetEntry(i), 0);

         EXPECT_EQ(x, -1 * y);
         if (!activeCuts(x, y)) {
            EXPECT_EQ(x, X_t{});
            EXPECT_EQ(y, Y_t{});
         }
      }

      tree.ResetBranchAddresses();
   }
}

TEST(RDFVarySnapshot, SimpleRDFWithFilters)
{
   constexpr auto filename = "VarySnapshot.root";
   RemoveFileRAII(filename);
   constexpr unsigned int N = 10;
   ROOT::RDF::RSnapshotOptions options;
   options.fLazy = true;
   options.fOverwriteIfExists = true;
   options.fIncludeVariations = true;

   auto cuts = [](float x, double y) { return x < 50 || y < -70; };

   auto h = ROOT::RDataFrame(N)
               .Define("entry_beforeX", [](ULong64_t e) { return e; }, {"rdfentry_"})
               .Define("x", [](ULong64_t e) -> float { return 10.f * e; }, {"rdfentry_"})
               .Define("entry_afterX", "rdfentry_")
               .Vary(
                  "x", [](float x) { return ROOT::RVecF{x - 0.5f, x + 0.5f}; }, {"x"}, 2, "xVar")
               .Define("y", [](float x) -> double { return -1. * x; }, {"x"})
               .Define("entry_afterVary", [](ULong64_t e) { return e; }, {"rdfentry_"})
               .Filter(cuts, {"x", "y"})
               .Snapshot("t", filename, {"x", "y", "entry_beforeX", "entry_afterX", "entry_afterVary"}, options);
   h.GetPtr();

   {
      std::unique_ptr<TFile> file{TFile::Open(filename)};
      auto tree = file->Get<TTree>("t");
      if (verbose)
         printTree(*tree);

      EXPECT_EQ(tree->GetEntries(), 9);    // Event number 6 won't pass any filters
      EXPECT_EQ(tree->GetNbranches(), 10); // 6 branches for x/y with variations, three not-varied branches, bitmask
      for (const auto branchName : {"x", "y", "x__xVar_0", "x__xVar_1", "y__xVar_0", "y__xVar_0"})
         EXPECT_NE(tree->FindBranch(branchName), nullptr) << branchName;

      checkOutput<float, double>(*tree, std::vector<std::string>{"__xVar_0", "__xVar_1"}, cuts);

      if (HasFailure()) {
         tree->Print();
         tree->Scan();
      }
   }
}

TEST(RDFVarySnapshot, RDFFromTTree)
{
   constexpr auto inFile = "VarySnapshot_1.root";
   constexpr auto outfile = "VarySnapshot_2.root";
   RemoveFileRAII(inFile, outfile);
   constexpr unsigned int N = 10;
   auto in = ROOT::RDataFrame(N)
                .Define("entry", [](ULong64_t e) { return e; }, {"rdfentry_"})
                .Define("x", [](ULong64_t e) -> int { return 10 * int(e); }, {"rdfentry_"})
                .Define("y", [](int x) -> float { return -1 * x; }, {"x"})
                .Snapshot("t", inFile, {"x", "y", "entry"});
   auto nextRDF = in.GetValue();
   {
      std::unique_ptr<TFile> fileIn{TFile::Open(inFile, "READ")};
      TTree *tree = fileIn->Get<TTree>("t");
      TBranch *branch = nullptr;

      ASSERT_NE(branch = tree->GetBranch("x"), nullptr);
      EXPECT_STREQ(static_cast<TLeaf *>(branch->GetListOfLeaves()->At(0))->GetTypeName(), "Int_t");

      ASSERT_NE(branch = tree->GetBranch("y"), nullptr);
      EXPECT_STREQ(static_cast<TLeaf *>(branch->GetListOfLeaves()->At(0))->GetTypeName(), "Float_t");
      if (verbose)
         tree->Scan("*", "", "", 5);
   }

   ROOT::RDF::RSnapshotOptions options;
   options.fLazy = true;
   options.fIncludeVariations = true;
   auto filter = [](int x, Long64_t y) { return (20 <= x && x < 70) && y != -50; };

   auto var = ROOT::RDataFrame("t", {inFile, inFile})
                 .Vary(
                    "x", [](int x) { return ROOT::RVecI{x - 1, x + 1}; }, {"x"}, 2, "xVariation")
                 .Redefine("y", [](int x) -> Long64_t { return -1 * x; }, {"x"})
                 .Filter(filter, {"x", "y"}, "nominal filter")
                 .Snapshot("t", outfile, {"x", "y", "entry"}, options);
   auto thirdRDF = var.GetValue();

   {
      std::unique_ptr<TFile> file{TFile::Open(outfile)};
      auto tree = file->Get<TTree>("t");
      if (verbose) {
         tree->Print();
         tree->Scan("*", "", "colsize=20", 20);
      }

      EXPECT_EQ(tree->GetEntries(), 2 * 6); // x=0, 10, 70, and y = -50 don't pass
      EXPECT_EQ(tree->GetNbranches(), 8);   // 3 variations of x and y, mask bits, and entry number
      for (const auto &[branchName, branchType] :
           std::initializer_list<std::pair<char const *, char const *>>{{"entry", "ULong64_t"},
                                                                        {"x", "Int_t"},
                                                                        {"y", "Long64_t"},
                                                                        {"x__xVariation_0", "Int_t"},
                                                                        {"x__xVariation_1", "Int_t"},
                                                                        {"y__xVariation_0", "Long64_t"},
                                                                        {"y__xVariation_1", "Long64_t"}}) {
         TBranch *branch;
         ASSERT_NE(branch = tree->GetBranch(branchName), nullptr);
         EXPECT_STREQ(static_cast<TLeaf *>(branch->GetListOfLeaves()->At(0))->GetTypeName(), branchType);
      }

      checkOutput<int, Long64_t>(*tree, std::vector<std::string>{"__xVariation_0", "__xVariation_1"}, filter);

      if (HasFailure()) {
         tree->Print();
         tree->Scan("*", "", "colsize=20", 10);
      }
   }
}

TEST(RDFVarySnapshot, Bitmask)
{
   constexpr auto filename = "VarySnapshot_bitmask.root";
   RemoveFileRAII(filename);
   std::string const treename = "testTree";
   constexpr unsigned int N = 15;
   constexpr unsigned int NSystematics = 130; // Will use three bitmask branches
   ROOT::RDF::RSnapshotOptions options;
   options.fLazy = false;
   options.fOverwriteIfExists = true;
   options.fIncludeVariations = true;

   auto cuts = [](int x, int y) { return x % 2 == 0 && y % 3 == 0; };

   ROOT::RDataFrame(N)
      .Define("x", [](ULong64_t e) -> int { return e; }, {"rdfentry_"})
      .Define("entry", [](ULong64_t e) { return e; }, {"rdfentry_"})
      .Vary(
         "x",
         [=](int x) {
            std::vector<int> systOffsets(NSystematics);
            std::iota(systOffsets.begin(), systOffsets.end(), 0);
            std::transform(systOffsets.begin(), systOffsets.end(), systOffsets.begin(),
                           [x](int offset) { return x + offset; });
            return ROOT::RVecI{systOffsets};
         },
         {"x"}, NSystematics, "xVar")
      .Define("y", [](int x) -> int { return -1 * x; }, {"x"})
      .Filter(cuts, {"x", "y"})
      .Snapshot(treename, filename, {"entry", "x", "y"}, options);

   {
      SCOPED_TRACE("Direct read of bitmask");
      TFile file(filename, "READ");
      std::unique_ptr<TTree> tree{file.Get<TTree>(treename.data())};
      ASSERT_NE(tree, nullptr);
      EXPECT_EQ(tree->GetEntries(), N);
      TBranch *branch = tree->GetBranch(("R_rdf_mask_" + treename + "_0").c_str());
      ASSERT_NE(branch, nullptr);

      auto *branchToIndexMap = file.Get<std::unordered_map<std::string, std::pair<std::string, unsigned int>>>(
         ("R_rdf_column_to_bitmask_mapping_" + treename).c_str());
      ASSERT_NE(branchToIndexMap, nullptr);
      for (const auto branchName : {"x", "y", "x__xVar_0", "x__xVar_1", "y__xVar_0", "y__xVar_0"}) {
         ASSERT_NE(branchToIndexMap->find(branchName), branchToIndexMap->end());
      }
      EXPECT_EQ((*branchToIndexMap)["x"], (*branchToIndexMap)["y"]);

      for (unsigned int systematic = 0; systematic < NSystematics; ++systematic) {
         int x, y;
         ULong64_t rdfentry;
         uint64_t bitmask;
         std::string const systematicName = std::string{"__xVar_"} + std::to_string(systematic);
         std::string const branchName_x = "x" + systematicName;
         std::string const branchName_y = "y" + systematicName;
         ASSERT_EQ(tree->SetBranchAddress(branchName_x.c_str(), &x), TTree::kMatch);
         ASSERT_EQ(tree->SetBranchAddress(branchName_y.c_str(), &y), TTree::kMatch);
         ASSERT_EQ(tree->SetBranchAddress("entry", &rdfentry), TTree::kMatch);
         const auto it = branchToIndexMap->find(branchName_x);
         ASSERT_NE(it, branchToIndexMap->end());
         ASSERT_EQ(tree->SetBranchAddress(it->second.first.c_str(), &bitmask), TTree::kMatch) << it->second.first;
         ASSERT_NE(branchToIndexMap->find(branchName_y), branchToIndexMap->end());
         EXPECT_EQ(it->second.first, (*branchToIndexMap)[branchName_y].first);
         EXPECT_EQ(it->second.second, (*branchToIndexMap)[branchName_y].second);
         for (unsigned int i = 0; i < N; ++i) {
            tree->GetEntry(i);
            EXPECT_EQ(rdfentry, i);
            const int xExpected = rdfentry + systematic;
            const int yExpected = -1 * xExpected;
            if (cuts(xExpected, yExpected)) {
               EXPECT_EQ(x, xExpected) << "event=" << i << " systematic=" << systematic << " rdfentry=" << rdfentry;
               EXPECT_EQ(y, yExpected) << "event=" << i << " systematic=" << systematic << " rdfentry=" << rdfentry;
            }

            const std::bitset<64> bs{bitmask};
            const auto bitIndex = it->second.second;
            EXPECT_EQ(cuts(xExpected, yExpected), bs[bitIndex])
               << "event=" << i << " syst=" << systematic << " x(" << branchName_x << ")=" << x << " y(" << branchName_y
               << ")=" << y << " rdfentry=" << rdfentry << " bitset: " << bs << " bitIndex: " << bitIndex;
            if (HasFailure())
               break;
         }
         tree->ResetBranchAddresses();
         if (HasFailure())
            break;
      }

      // Test that the Masked column reader works
      {
         SCOPED_TRACE("Usage of bitmask in RDF");
         auto rdf = ROOT::RDataFrame(treename, filename);

         auto filterAvailable = rdf.FilterAvailable("x");
         auto meanX = filterAvailable.Mean<int>("x");
         auto meanY = filterAvailable.Mean<int>("y");
         auto count = filterAvailable.Count();

         EXPECT_EQ(count.GetValue(), 3ull); // 0, 6, 12
         EXPECT_EQ(meanX.GetValue(), 6.);
         EXPECT_EQ(meanY.GetValue(), -6.);

         // Test reading invalid columns
         auto mean = rdf.Mean<int>("x");
         EXPECT_THROW(mean.GetValue(), std::out_of_range);

         for (unsigned int systematicIndex : {0, 1, 100}) {
            const std::string systematic = "__xVar_" + std::to_string(systematicIndex);
            auto filterAv = rdf.FilterAvailable("x" + systematic);
            auto meanX_sys = filterAv.Mean("x" + systematic);
            auto meanY_sys = filterAv.Mean("y" + systematic);
            auto count_sys = filterAv.Count();

            std::vector<int> expect(N);
            std::iota(expect.begin(), expect.end(), systematicIndex);

            const auto nVal = std::count_if(expect.begin(), expect.end(), [](int v) { return v % 6 == 0; });
            // gcc8.5 on alma8 doesn't support transform_reduce, nor reduce
            // const int sum = std::transform_reduce(expect.begin(), expect.end(), 0, std::plus<>(),
            //                                          [](int v) { return v % 6 == 0 ? v : 0; });
            std::transform(expect.begin(), expect.end(), expect.begin(), [](int v) { return v % 6 == 0 ? v : 0; });
            const int sum = std::accumulate(expect.begin(), expect.end(), 0);

            ASSERT_EQ(count_sys.GetValue(), nVal) << "systematic: " << systematic;
            EXPECT_EQ(meanX_sys.GetValue(), sum / nVal) << "systematic: " << systematic;
            EXPECT_EQ(meanY_sys.GetValue(), -1. * sum / nVal) << "systematic: " << systematic;
         }
      }

      if (HasFailure()) {
         tree->Scan("entry:x:y:x__xVar_0:y__xVar_0:x__xVar_1:y__xVar_1:x__xVar_2:y__xVar_2");
      }
   }
}

TEST(RDFVarySnapshot, TwoVariationsInSameFile)
{
   constexpr auto filename = "VarySnapshot_bitmask_twoInSameFile.root";
   RemoveFileRAII(filename);
   std::string const treename1 = "testTree1";
   std::string const treename2 = "testTree2";
   constexpr unsigned int N = 20;
   ROOT::RDF::RSnapshotOptions options;
   options.fOverwriteIfExists = true;
   options.fIncludeVariations = true;
   options.fLazy = false;

   auto rdf = ROOT::RDataFrame(N)
                 .Define("x", [](ULong64_t e) -> int { return e; }, {"rdfentry_"})
                 .Vary(
                    "x", [](int x) { return ROOT::RVecI{x - 1, x + 1}; }, {"x"}, 2, "xVar")
                 .Define("y", [](int x) -> int { return -1 * x; }, {"x"});

   auto cuts1 = [](int x, int y) { return x % 2 == 0 && abs(y) <= 10; };
   auto snap1 = rdf.Filter(cuts1, {"x", "y"}).Snapshot(treename1, filename, {"x", "y"}, options);

   options.fOverwriteIfExists = false;
   options.fMode = "UPDATE";
   auto cuts2 = [](int x, int y) { return abs(x % 2) == 1 && abs(y) <= 10; };
   auto snap2 = rdf.Filter(cuts2, {"x", "y"}).Snapshot(treename2, filename, {"x", "y"}, options);

   std::unique_ptr<TFile> file{TFile::Open(filename)};
   EXPECT_NE(file->GetKey(("R_rdf_column_to_bitmask_mapping_" + treename1).c_str()), nullptr);
   EXPECT_NE(file->GetKey(("R_rdf_column_to_bitmask_mapping_" + treename2).c_str()), nullptr);

   // In Windows, an exception is thrown as expected, but it cannot be caught for the time being:
#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
   EXPECT_THROW(ROOT::RDataFrame(treename1, filename).Mean("x").GetValue(), std::out_of_range);
   EXPECT_THROW(ROOT::RDataFrame(treename2, filename).Mean("x").GetValue(), std::out_of_range);
   EXPECT_THROW(ROOT::RDataFrame(treename1, filename).FilterAvailable("x__xVar_0").Mean("x").GetValue(),
                std::out_of_range);
   EXPECT_THROW(ROOT::RDataFrame(treename2, filename).FilterAvailable("x__xVar_1").Mean("x").GetValue(),
                std::out_of_range);
#endif

   const std::vector<int> evens{0, 2, 4, 6, 8, 10};
   const std::vector<int> odds{1, 3, 5, 7, 9};
   EXPECT_EQ(ROOT::RDataFrame(treename1, filename).FilterAvailable("x").Take<int>("x").GetValue(), evens);
   EXPECT_EQ(ROOT::RDataFrame(treename2, filename).FilterAvailable("x").Take<int>("x").GetValue(), odds);
   const std::vector<int> evensVarDo{0, 2, 4, 6, 8, 10};
   const std::vector<int> oddsVarDo{-1, 1, 3, 5, 7, 9};
   EXPECT_EQ(ROOT::RDataFrame(treename1, filename).FilterAvailable("x__xVar_0").Take<int>("x__xVar_0").GetValue(),
             evensVarDo);
   EXPECT_EQ(ROOT::RDataFrame(treename2, filename).FilterAvailable("x__xVar_0").Take<int>("x__xVar_0").GetValue(),
             oddsVarDo);
   const std::vector<int> evensVarUp{2, 4, 6, 8, 10};
   const std::vector<int> oddsVarUp{1, 3, 5, 7, 9};
   EXPECT_EQ(ROOT::RDataFrame(treename1, filename).FilterAvailable("x__xVar_1").Take<int>("x__xVar_1").GetValue(),
             evensVarUp);
   EXPECT_EQ(ROOT::RDataFrame(treename2, filename).FilterAvailable("x__xVar_1").Take<int>("x__xVar_1").GetValue(),
             oddsVarUp);
}

TEST(RDFVarySnapshot, SnapshotCollections)
{
   constexpr auto filename = "VarySnapshot_collections.root";
   RemoveFileRAII(filename);
   constexpr unsigned int N = 10;
   ROOT::RDF::RSnapshotOptions options;
   options.fIncludeVariations = true;

   gInterpreter->GenerateDictionary("std::vector<ROOT::VecOps::RVec<int> >", "vector;ROOT/RVec.hxx");

   auto cuts = [](int x, ROOT::RVecI const &y) { return x % 2 == 0 || y[0] == 5; };

   auto variation = ROOT::RDataFrame(N)
                       .Define("x", [](ULong64_t e) -> int { return int(e); }, {"rdfentry_"})
                       .Define("entry", [](ULong64_t e) { return e; }, {"rdfentry_"})
                       .Define("dummyBranch", [](ULong64_t e) { return e; }, {"rdfentry_"})
                       .Vary(
                          "x", [](int x) { return ROOT::RVecI{x + 1, x * 3}; }, {"x"}, 2, "xVariation")
                       .Define("y", [](int x) { return ROOT::RVecI{x, x + 1, x + 2, x + 3}; }, {"x"})
                       .Filter(cuts, {"x", "y"})
                       .Snapshot("t", filename, {"x", "y", "entry"}, options);
   variation.GetPtr();

   std::unique_ptr<TFile> file{TFile::Open(filename)};
   auto tree = file->Get<TTree>("t");
   if (verbose)
      printTree(*tree);

   EXPECT_EQ(tree->GetEntries(), N);
   EXPECT_EQ(tree->GetNbranches(), 8);
   for (const auto branchName :
        {"x", "y", "x__xVariation_0", "x__xVariation_0", "y__xVariation_0", "y__xVariation_0", "entry"})
      EXPECT_NE(tree->GetBranch(branchName), nullptr) << branchName;
   EXPECT_EQ(tree->GetBranch("dummyBranch"), nullptr);

   for (std::string systematicName : {"", "__xVariation_0", "__xVariation_1"}) {
      ULong64_t entry;
      int x;
      ROOT::RVecI *y = nullptr;
      ASSERT_EQ(tree->SetBranchAddress<ULong64_t>("entry", &entry), TTree::kMatch);
      ASSERT_EQ(tree->SetBranchAddress<int>(("x" + systematicName).c_str(), &x), TTree::kMatch)
         << ("x" + systematicName);
      ASSERT_EQ(tree->SetBranchAddress(("y" + systematicName).c_str(), &y), TTree::kMatch) << ("y" + systematicName);

      for (unsigned int event = 0; event < N; ++event) {
         ASSERT_GT(tree->GetEntry(event), 0);
         const auto originalX =
            (systematicName == "")
               ? entry
               : ((systematicName.find("xVariation_0") != std::string::npos) ? entry + 1 : entry * 3);
         const bool passCuts = (originalX % 2 == 0) || originalX == 5;

         if (passCuts)
            EXPECT_EQ(x, originalX) << "sys:'" << systematicName << "' originalX: " << originalX << " event: " << event;
         else
            EXPECT_EQ(x, 0) << "sys:'" << systematicName << "' originalX: " << originalX << " event: " << event;

         ASSERT_EQ(y->size(), passCuts ? 4 : 0)
            << "sys:'" << systematicName << "' entry: " << entry << " originalX: " << originalX << " event: " << event;
         for (unsigned int i = 0; i < y->size(); ++i) {
            EXPECT_EQ((*y)[i], x + i);
         }
      }
      tree->ResetBranchAddresses();
   }
}

TEST(RDFVarySnapshot, SnapshotVirtualClass)
{
   constexpr auto filename = "VarySnapshot_virtualClass.root";
   RemoveFileRAII(filename);
   ROOT::RDataFrame rdf(10);
   ROOT::RDF::RSnapshotOptions opt;
   opt.fIncludeVariations = true;

   auto result =
      rdf.Define("tf", [](ULong64_t entry) { return TwoFloats_Snapshot(entry); }, {"rdfentry_"})
         .Vary("tf",
               [](TwoFloats_Snapshot const &tf) {
                  TwoFloats_Snapshot up, down;
                  up.a = tf.a;
                  up.b = tf.a + 1.f;
                  down.a = tf.a;
                  down.b = tf.a - 1.f;
                  return ROOT::RVec<TwoFloats_Snapshot>{up, down};
               },
               {"tf"}, {"Up", "Down"})
         .Filter(
            [](TwoFloats_Snapshot const &tf) { return 2. * tf.a == tf.b || static_cast<unsigned int>(tf.a) % 2 == 0; },
            {"tf"})
         .Snapshot("t", filename, {"tf", "rdfentry_"}, opt);

   {
      TFile file(filename, "READ");
      std::unique_ptr<TTree> tree{file.Get<TTree>("t")};
      ASSERT_TRUE(tree);
      if (verbose) {
         tree->Print("");
         tree->Scan("*", "", "colsize=20");
      }

      TwoFloats_Snapshot *tf = nullptr, *tf_Up = nullptr, *tf_Down = nullptr;
      ASSERT_EQ(tree->SetBranchAddress<TwoFloats_Snapshot>("tf", &tf), TTree::kMatch);
      ASSERT_EQ(tree->SetBranchAddress<TwoFloats_Snapshot>("tf__tf_Up", &tf_Up), TTree::kMatch);
      ASSERT_EQ(tree->SetBranchAddress<TwoFloats_Snapshot>("tf__tf_Down", &tf_Down), TTree::kMatch);
      ASSERT_EQ(tree->GetEntries(), 10);
      for (unsigned int i = 0; i < 10; ++i) {
         tree->GetEntry(i);
         EXPECT_EQ(tf->a, i);
         EXPECT_EQ(tf->b, 2 * i);
         if (i == 1) {
            // Passes because a == 1 && b == 2 both for b=2*a and b=a+1
            EXPECT_EQ(tf_Up->a, 1.f);
            EXPECT_EQ(tf_Up->b, 2.f);
         }
         if (i % 2 == 0) {
            EXPECT_EQ(tf_Up->a, i);
            EXPECT_EQ(tf_Up->b, i + 1.f);
            EXPECT_EQ(tf_Down->a, i);
            EXPECT_EQ(tf_Down->b, i - 1.f);
         }
      }
   }
}

// https://github.com/root-project/root/issues/20320
TEST(RDFVarySnapshot, GH20320)
{
   const char *fileName{"dataframe_snapshot_with_variations_regression_gh20330.root"};
   RemoveFileRAII(fileName);

   ROOT::RDataFrame df{1};

   auto df_def = df.Define("val", []() { return 2; });

   auto df_var =
      df_def.Vary("val", [](int val) { return ROOT::RVecI{val - 1, val + 1}; }, {"val"}, {"down", "up"}, "var");

   // Jitted filters used to break the Snapshot because:
   // - It did not JIT the RJittedFilter before requesting for the varied filters
   // - It did not take into account that the previous nodes of the Snapshot could be of different types
   auto df_fil = df_var.Filter("val > 0");

   ROOT::RDF::RSnapshotOptions opts;
   opts.fIncludeVariations = true;
   auto snap = df_fil.Snapshot("tree", fileName, {"val"}, opts);

   auto take_val = snap->Take<int>("val");
   auto take_var_up = snap->Take<int>("val__var_up");
   auto take_var_down = snap->Take<int>("val__var_down");

   EXPECT_EQ(take_val->size(), 1);
   EXPECT_EQ(take_var_up->size(), 1);
   EXPECT_EQ(take_var_down->size(), 1);

   EXPECT_EQ(take_var_down->front(), 1);
   EXPECT_EQ(take_val->front(), 2);
   EXPECT_EQ(take_var_up->front(), 3);
}

// https://github.com/root-project/root/issues/20506
// When snapshot is invoked on JITted Defines with variations, the Snapshot
// Action needs to JIT the computation graph before creating outputs.
// When this wasn't happening, the varied Defines be identical to nominal and
// not be snapshot.
TEST(RDFVarySnapshot, IncludeDependentColumns_JIT)
{
   const char *fileName{"dataframe_snapshot_include_dependent_columns.root"};
   RemoveFileRAII(fileName);
   constexpr unsigned int N = 10;

   ROOT::RDataFrame rdf(N);
   auto df = rdf.Define("Muon_pt", "return static_cast<double>(rdfentry_)")
                .Vary("Muon_pt", "ROOT::VecOps::RVec<double>({Muon_pt*0.5, Muon_pt*2})", {"down", "up"}, "muon_unc")
                .Define("Muon_2pt", "return 2. * Muon_pt;");

   ROOT::RDF::RSnapshotOptions opts;
   opts.fOverwriteIfExists = true;
   opts.fIncludeVariations = true;

   auto snapshot = df.Snapshot("Events", fileName, {"Muon_pt", "Muon_2pt"}, opts);

   TFile file(fileName);
   auto tree = file.Get<TTree>("Events");
   ASSERT_NE(tree, nullptr);
   tree->Scan();

   double Muon_pt, Muon_pt_up, Muon_pt_down;
   double Muon_2pt, Muon_2pt_up, Muon_2pt_down;
   ASSERT_EQ(TTree::kMatch, tree->SetBranchAddress("Muon_pt", &Muon_pt));
   ASSERT_EQ(TTree::kMatch, tree->SetBranchAddress("Muon_pt__muon_unc_up", &Muon_pt_up));
   ASSERT_EQ(TTree::kMatch, tree->SetBranchAddress("Muon_pt__muon_unc_down", &Muon_pt_down));
   ASSERT_EQ(TTree::kMatch, tree->SetBranchAddress("Muon_2pt", &Muon_2pt));
   ASSERT_EQ(TTree::kMatch, tree->SetBranchAddress("Muon_2pt__muon_unc_up", &Muon_2pt_up));
   ASSERT_EQ(TTree::kMatch, tree->SetBranchAddress("Muon_2pt__muon_unc_down", &Muon_2pt_down));

   EXPECT_EQ(tree->GetEntries(), N);
   for (unsigned int i = 0; i < tree->GetEntries(); ++i) {
      ASSERT_GT(tree->GetEntry(i), 0);
      EXPECT_EQ(2. * Muon_pt, Muon_2pt);

      EXPECT_EQ(0.5 * Muon_pt, Muon_pt_down);
      EXPECT_EQ(0.5 * Muon_2pt, Muon_2pt_down);
      EXPECT_EQ(2. * Muon_pt, Muon_pt_up);
      EXPECT_EQ(2. * Muon_2pt, Muon_2pt_up);
   }
}
