/// \file
/// \ingroup tutorial_dataframe
/// \notebook
/// Fill and draw a bar chart from a categorical/string column with RDataFrame.
///
/// RDataFrame has no dedicated action for categorical/string columns, but the
/// generic RDataFrame::Fill() action can fill any user-provided object that
/// exposes Fill() and Merge() methods. This shows how to use it to build and
/// draw a bar chart from a string column, reusing the "Nation" column of the
/// classic CERN staff dataset also used in hist006_TH1_bar_charts.C.
///
/// Classic ROOT trees often store short strings in fixed-size char[] branches
/// (leaf type "C"), which RDataFrame reads as ROOT::VecOps::RVec<char> rather
/// than std::string. A one-line Define() converts such a column into a real
/// std::string column before filling.
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
   TH1D h;

   AlphaNumHist(const char *name, const char *title) : h(name, title, 1, 0, 1)
   {
      h.GetXaxis()->SetAlphanumeric(true);
      h.SetCanExtend(TH1::kAllAxes);
   }

   void Fill(const std::string &s) { h.Fill(s.c_str(), 1.); }

   // Required so RDataFrame can merge partial per-slot results when run with
   // implicit multi-threading enabled.
   void Merge(const std::vector<AlphaNumHist *> &others)
   {
      TList l;
      for (auto *o : others)
         l.Add(&o->h);
      h.Merge(&l);
   }
};

void df041_alphanumericHistograms()
{
   // Reuse the same "cernstaff.root" dataset as hist006_TH1_bar_charts.C,
   // generating it first if it doesn't already exist.
   TString filedir = gROOT->GetTutorialDir();
   filedir += TString("/io/tree/");
   TString filename = "cernstaff.root";
   bool fileNotFound = gSystem->AccessPathName(filename);
   if (fileNotFound) {
      TString macroName = filedir + "tree500_cernbuild.C";
      if (!gInterpreter->IsLoaded(macroName))
         gInterpreter->LoadMacro(macroName);
      gROOT->ProcessLineFast("tree500_cernbuild()");
   }

   ROOT::RDataFrame df0("T", filename.Data());

   // "Nation" is a classic char[] branch, read by RDataFrame as
   // ROOT::VecOps::RVec<char>. Converting it to a real std::string column
   // makes it fillable through the AlphaNumHist adapter above.
   auto df = df0.Define("NationStr", [](const ROOT::VecOps::RVec<char> &c) { return std::string(c.begin(), c.end()); },
                        {"Nation"});

   AlphaNumHist model("hNation", "Staff by nation;Nation;Count");
   auto h = df.Fill<std::string>(model, {"NationStr"});
   h->h.LabelsDeflate("X");

   auto c1 = new TCanvas("c1", "Bar chart from RDataFrame", 700, 500);
   h->h.SetFillColor(45);
   h->h.DrawCopy("bar2");
}
