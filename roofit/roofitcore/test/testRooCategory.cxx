// Tests for the RooCategory
// Author: Jonas Rembser, CERN  04/2021

#include <Roo1DTable.h>
#include <RooBinningCategory.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooMappedCategory.h>
#include <RooMultiCategory.h>
#include <RooRealVar.h>
#include <RooSuperCategory.h>
#include <RooThresholdCategory.h>

#include <TTree.h>

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

// GitHub issue 10278: RooDataSet incorrectly loads RooCategory values from TTree branch of type Short_t
TEST(RooCategory, CategoryDefineMultiState)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   TTree tree("test_tree", "Test tree");
   Short_t cat_in;
   tree.Branch("cat", &cat_in);

   cat_in = 2; // category B
   tree.Fill();

   RooCategory cat("cat", "Category", {{"B_cat", 2}, {"A_cat", 3}});
   RooDataSet data("data", "RooDataSet", RooArgSet(cat), RooFit::Import(tree));

   EXPECT_EQ(static_cast<RooCategory &>((*data.get(0))["cat"]).getCurrentIndex(), 2);
}

/// Roo1DTable tabulation of category contents of a dataset, and named ranges
/// on categories used in dataset reduction. The tables are checked against
/// counts computed while deterministically filling the dataset. Replaces the
/// former stressRooFit test based on the rf404 tutorial, which compared
/// against stored reference tables.
TEST(RooCategory, TablesAndRanges)
{
   RooCategory tagCat("tagCat", "Tagging category");
   tagCat.defineType("Lepton");
   tagCat.defineType("Kaon");
   tagCat.defineType("NetTagger-1");
   tagCat.defineType("NetTagger-2");

   RooCategory b0flav("b0flav", "B0 flavour eigenstate");
   b0flav.defineType("B0", -1);
   b0flav.defineType("B0bar", 1);

   RooRealVar x("x", "x", 0, 10);
   RooDataSet data("data", "data", {x, b0flav, tagCat});

   const std::vector<std::string> tagStates{"Lepton", "Kaon", "NetTagger-1", "NetTagger-2"};

   // Fill the dataset deterministically and keep track of the expected counts
   std::map<std::string, int> nFlav;
   std::map<std::string, int> nTagWithCut;
   std::map<std::string, int> nComb;
   std::map<std::string, int> nGood;
   int nCutTotal = 0;
   for (int i = 0; i < 10000; ++i) {
      const std::string &tag = tagStates[i % 4];
      const std::string flav = ((i / 2) % 2) ? "B0bar" : "B0";
      const double xVal = (i % 100) / 10.0;
      x.setVal(xVal);
      tagCat.setLabel(tag.c_str());
      b0flav.setLabel(flav.c_str());
      data.add({x, b0flav, tagCat});

      ++nFlav[flav];
      ++nComb["{" + tag + ";" + flav + "}"];
      if (xVal > 8.23) {
         ++nTagWithCut[tag];
         ++nCutTotal;
      }
      if (tag == "Lepton" || tag == "Kaon") {
         ++nGood[tag];
      }
   }

   // Table of a single category
   std::unique_ptr<Roo1DTable> btable{data.table(b0flav)};
   EXPECT_DOUBLE_EQ(btable->get("B0"), nFlav["B0"]);
   EXPECT_DOUBLE_EQ(btable->get("B0bar"), nFlav["B0bar"]);

   // Table for the subset of events matching a cut expression
   std::unique_ptr<Roo1DTable> ttable{data.table(tagCat, "x>8.23")};
   for (const auto &tag : tagStates) {
      EXPECT_DOUBLE_EQ(ttable->get(tag.c_str()), nTagWithCut[tag]) << tag;
      EXPECT_DOUBLE_EQ(ttable->getFrac(tag.c_str()), double(nTagWithCut[tag]) / nCutTotal) << tag;
   }

   // Table of all state combinations of two categories
   std::unique_ptr<Roo1DTable> bttable{data.table({tagCat, b0flav})};
   for (const auto &[label, count] : nComb) {
      EXPECT_DOUBLE_EQ(bttable->get(label.c_str()), count) << label;
   }

   // Named category ranges used in dataset reduction
   tagCat.setRange("good", "Lepton,Kaon");
   tagCat.addToRange("soso", "NetTagger-1");
   tagCat.addToRange("soso", "NetTagger-2");

   std::unique_ptr<RooAbsData> goodData{data.reduce(RooFit::CutRange("good"))};
   EXPECT_EQ(goodData->numEntries(), nGood["Lepton"] + nGood["Kaon"]);
   std::unique_ptr<Roo1DTable> gtable{goodData->table(tagCat)};
   EXPECT_DOUBLE_EQ(gtable->get("Lepton"), nGood["Lepton"]);
   EXPECT_DOUBLE_EQ(gtable->get("Kaon"), nGood["Kaon"]);
   EXPECT_DOUBLE_EQ(gtable->get("NetTagger-1"), 0.);
   EXPECT_DOUBLE_EQ(gtable->get("NetTagger-2"), 0.);

   std::unique_ptr<RooAbsData> sosoData{data.reduce(RooFit::CutRange("soso"))};
   EXPECT_EQ(sosoData->numEntries(), 10000 - nGood["Lepton"] - nGood["Kaon"]);
}

/// Real-to-category mapping functions RooThresholdCategory and
/// RooBinningCategory, checked entry-by-entry against the expected mapping.
/// Replaces the former stressRooFit test based on the rf405 tutorial, which
/// compared against stored reference tables and plots.
TEST(RooCategory, RealToCategoryFunctions)
{
   RooRealVar x("x", "x", 0, 10);

   // Threshold mapping: each threshold assigns all values below it (and above
   // any lower threshold) to the given state
   RooThresholdCategory xRegion("xRegion", "region of x", x, "Background");
   xRegion.addThreshold(4.23, "Background");
   xRegion.addThreshold(5.23, "SideBand");
   xRegion.addThreshold(8.23, "Signal");
   xRegion.addThreshold(9.23, "SideBand");

   // Binning mapping based on a named binning
   x.setBins(10, "coarse");
   RooBinningCategory xBins("xBins", "coarse bins in x", x, "coarse");

   auto expectedRegion = [](double v) -> std::string {
      if (v < 4.23)
         return "Background";
      if (v < 5.23)
         return "SideBand";
      if (v < 8.23)
         return "Signal";
      if (v < 9.23)
         return "SideBand";
      return "Background";
   };

   RooDataSet data("data", "data", x);

   std::map<std::string, int> nRegion;
   std::map<std::string, int> nBin;
   int nAltFirstHalf = 0;
   const int nEntries = 1000;
   for (int i = 0; i < nEntries; ++i) {
      // Values chosen such that they never coincide with a threshold or bin
      // boundary, where the expected mapping would be floating-point fragile
      const double v = 10. * (i + 0.5) / nEntries;
      x.setVal(v);

      const std::string region = expectedRegion(v);
      const std::string bin = "x_coarse_bin" + std::to_string(int(v));
      EXPECT_STREQ(xRegion.getCurrentLabel(), region.c_str()) << "x = " << v;
      EXPECT_STREQ(xBins.getCurrentLabel(), bin.c_str()) << "x = " << v;

      data.add(x);
      ++nRegion[region];
      ++nBin[bin];
      if (int(v) % 2 == 1 && i < nEntries / 2) {
         ++nAltFirstHalf;
      }
   }

   // Add the category functions as columns to the dataset and cross-check the
   // stored values entry-by-entry
   auto *xr = static_cast<RooCategory *>(data.addColumn(xRegion));
   auto *xb = static_cast<RooCategory *>(data.addColumn(xBins));
   for (int i = 0; i < nEntries; ++i) {
      const RooArgSet *row = data.get(i);
      const double v = row->getRealValue("x");
      EXPECT_STREQ(row->getCatLabel("xRegion"), expectedRegion(v).c_str()) << "entry " << i;
      EXPECT_STREQ(row->getCatLabel("xBins"), ("x_coarse_bin" + std::to_string(int(v))).c_str()) << "entry " << i;
   }

   // Tabulate the computed columns
   std::unique_ptr<Roo1DTable> rtable{data.table(*xr)};
   for (const auto &[label, count] : nRegion) {
      EXPECT_DOUBLE_EQ(rtable->get(label.c_str()), count) << label;
   }
   std::unique_ptr<Roo1DTable> btable{data.table(*xb)};
   for (const auto &[label, count] : nBin) {
      EXPECT_DOUBLE_EQ(btable->get(label.c_str()), count) << label;
   }

   // Use a named range on the computed category column together with an event
   // range in a dataset reduction
   xb->setRange("alt", "x_coarse_bin1,x_coarse_bin3,x_coarse_bin5,x_coarse_bin7,x_coarse_bin9");
   std::unique_ptr<RooAbsData> dataSel{data.reduce(RooFit::CutRange("alt"), RooFit::EventRange(0, nEntries / 2))};
   EXPECT_EQ(dataSel->numEntries(), nAltFirstHalf);
}

/// Category-to-category mapping functions RooMappedCategory, RooSuperCategory
/// and RooMultiCategory, checked state-by-state against the expected mapping.
/// Replaces the former stressRooFit test based on the rf406 tutorial, which
/// compared against stored reference tables.
TEST(RooCategory, CategoryToCategoryFunctions)
{
   RooCategory tagCat("tagCat", "Tagging category");
   tagCat.defineType("Lepton");
   tagCat.defineType("Kaon");
   tagCat.defineType("NetTagger-1");
   tagCat.defineType("NetTagger-2");

   RooCategory b0flav("b0flav", "B0 flavour eigenstate");
   b0flav.defineType("B0", -1);
   b0flav.defineType("B0bar", 1);

   // Category-to-category mapping with explicit and wildcard expressions
   RooMappedCategory tcatType("tcatType", "tagCat type", tagCat, "Cut based");
   tcatType.map("Lepton", "Cut based");
   tcatType.map("Kaon", "Cut based");
   tcatType.map("NetTagger*", "Neural Network");

   auto expectedType = [](const std::string &tag) -> std::string {
      return (tag == "Lepton" || tag == "Kaon") ? "Cut based" : "Neural Network";
   };

   // Product categories of lvalue categories (super) and of any category
   // functions (multi)
   RooSuperCategory b0Xtcat("b0Xtcat", "b0flav X tagCat", {b0flav, tagCat});
   RooMultiCategory b0Xttype("b0Xttype", "b0flav X tagType", {b0flav, tcatType});

   const std::vector<std::string> tagStates{"Lepton", "Kaon", "NetTagger-1", "NetTagger-2"};
   for (const auto &flav : {"B0", "B0bar"}) {
      for (const auto &tag : tagStates) {
         b0flav.setLabel(flav);
         tagCat.setLabel(tag.c_str());
         EXPECT_STREQ(tcatType.getCurrentLabel(), expectedType(tag).c_str()) << tag;
         const std::string superLabel = "{" + std::string(flav) + ";" + tag + "}";
         EXPECT_STREQ(b0Xtcat.getCurrentLabel(), superLabel.c_str());
         const std::string multiLabel = "{" + std::string(flav) + ";" + expectedType(tag) + "}";
         EXPECT_STREQ(b0Xttype.getCurrentLabel(), multiLabel.c_str());
      }
   }

   // A super category is an lvalue: assigning a state must propagate to the
   // input categories
   EXPECT_FALSE(b0Xtcat.setLabel("{B0bar;Kaon}")); // returns true on error
   EXPECT_STREQ(b0flav.getCurrentLabel(), "B0bar");
   EXPECT_STREQ(tagCat.getCurrentLabel(), "Kaon");

   // Tabulate the mapped categories in a deterministically filled dataset
   RooDataSet data("data", "data", {b0flav, tagCat});
   std::map<std::string, int> nType;
   std::map<std::string, int> nSuper;
   for (int i = 0; i < 1000; ++i) {
      const std::string &tag = tagStates[i % 4];
      const std::string flav = ((i / 2) % 2) ? "B0bar" : "B0";
      tagCat.setLabel(tag.c_str());
      b0flav.setLabel(flav.c_str());
      data.add({b0flav, tagCat});
      ++nType[expectedType(tag)];
      ++nSuper["{" + flav + ";" + tag + "}"];
   }

   std::unique_ptr<Roo1DTable> mtable{data.table(tcatType)};
   EXPECT_DOUBLE_EQ(mtable->get("Cut based"), nType["Cut based"]);
   EXPECT_DOUBLE_EQ(mtable->get("Neural Network"), nType["Neural Network"]);

   std::unique_ptr<Roo1DTable> stable{data.table(b0Xtcat)};
   for (const auto &[label, count] : nSuper) {
      EXPECT_DOUBLE_EQ(stable->get(label.c_str()), count) << label;
   }
}
