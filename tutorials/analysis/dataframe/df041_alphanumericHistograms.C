/// \file
/// \ingroup tutorial_dataframe
/// \notebook
/// Fill and draw a bar chart from a categorical column with RDataFrame.
///
/// RDataFrame has no dedicated action for categorical/string columns, but the
/// generic RDataFrame::Fill() action can fill any user-provided object that
/// exposes Fill() and Merge() methods. This shows how to use it to build and
/// draw a bar chart from a string column.
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// \date July 2026
/// \author Mehkaan Khan

// A small adapter around TH1D that lets RDataFrame's generic Fill() action
// build a histogram from a std::string column: TH1D::Fill(const char*, Double_t)
// takes no default weight and std::string has no implicit conversion to
// const char*, so RDataFrame can't call it directly on a plain TH1D. Wrapping
// it in a class with our own Fill(const std::string&) sidesteps both issues.
struct AlphaNumHist {
   TH1D histo;

   AlphaNumHist(const char *name, const char *title) : histo(name, title, 1, 0, 1)
   {
      histo.GetXaxis()->SetAlphanumeric(true);
      histo.SetCanExtend(TH1::kAllAxes);
   }

   void Fill(const std::string &s) { histo.Fill(s.c_str(), 1.); }

   // Required so RDataFrame can merge partial per-slot results when run with
   // implicit multi-threading enabled.
   void Merge(const std::vector<AlphaNumHist *> &others)
   {
      TList l;
      for (auto *o : others)
         l.Add(&o->histo);
      histo.Merge(&l);
   }
};

void writeData(std::string_view datasetName, std::string_view fileName)
{
   std::random_device rd;
   std::mt19937 gen{rd()};
   std::uniform_int_distribution<int> distrib{0, 2};

   const std::vector<std::string> colours{"RED", "GREEN", "BLUE"};

   ROOT::RDataFrame df{100};
   df.Define("colour", [&]() { return colours[distrib(gen)]; }).Snapshot(datasetName, fileName);
}

auto canvas = std::make_unique<TCanvas>("c");

void df041_alphanumericHistograms()
{
   writeData("tree", "df041_alphanumericHistograms.root");

   ROOT::RDataFrame df("tree", "df041_alphanumericHistograms.root");

   AlphaNumHist model("hColour", "Entries by colour;Colour;Count");
   auto result = df.Fill<std::string>(model, {"colour"});
   result->histo.LabelsDeflate("X");

   result->histo.SetFillColor(45);
   result->histo.DrawCopy("bar2");
}
