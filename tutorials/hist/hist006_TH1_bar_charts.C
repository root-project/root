/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Draw 1D histograms as bar charts
///
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author Rene Brun

TCanvas *hist006_TH1_bar_charts()
{
   // Try to open first the file cernstaff.root in tutorials/tree directory
   TString filedir = gROOT->GetTutorialDir();
   filedir += TString("/tree/");
   TString filename = "cernstaff.root";
   // Note that `AccessPathName` returns 0 (false) on success!
   bool fileNotFound = gSystem->AccessPathName(filename); 

   // If the file is not found try to generate it using the macro tree/cernbuild.C
   if (fileNotFound) {
      TString macroName = filedir + "cernbuild.C";
      if (!gInterpreter->IsLoaded(macroName)) gInterpreter->LoadMacro(macroName);
      gROOT->ProcessLineFast("cernbuild()");
   }

   auto file = std::unique_ptr<TFile>(TFile::Open(filename, "READ"));
   if (!file) {
      Error("hbars", "file cernstaff.root not found");
      return nullptr;
   }

   // Retrieve the TTree named "T" contained in the file
   auto tree = file->Get<TTree>("T");
   if (!tree) {
      Error("hbars", "Tree T is not present in file %s", file->GetName());
      return nullptr;
   }
   tree->SetFillColor(45);

   // Create the canvas to draw on
   TCanvas *c1 = new TCanvas("c1","histograms with bars", 700, 800);
   c1->SetFillColor(42);
   // Divide it vertically in 2 sections
   int ndivsX = 1;
   int ndivsY = 2;
   c1->Divide(ndivsX, ndivsY);

   // Horizontal bar chart
   auto *curPad = c1->cd(1); // select top section. Section 1 is the first sub-section.
   curPad->SetGrid();
   curPad->SetLogx();
   curPad->SetFrameFillColor(33);
   // Use the "hbar2" option to draw the tree as a horizontal bar chart
   tree->Draw("Nation","","hbar2");

   // Vertical bar chart
   curPad = c1->cd(2);
   curPad->SetGrid();
   curPad->SetFrameFillColor(33);
   // This line makes the TTree draw its "Division" branch to a new histogram called "hDiv".
   // We use "goff" because we don't want to really draw it to screen but we are only interested
   // in generating the histogram from it (which we'll display ourselves later).
   tree->Draw("Division>>hDiv","","goff");
   // Retrieve the generated histogram
   TH1F *hDiv = file->Get<TH1F>("hDiv");
   hDiv->SetStats(0);
   // Clone the histogram into a new one called "hDivFR".
   TH1F *hDivFR = static_cast<TH1F*>(hDiv->Clone("hDivFR"));
   // Overwrite the contents of the newly-cloned histogram to only keep the entries matching our
   // selection (second argument of TTree::Draw()). 
   tree->Draw("Division>>hDivFR","Nation==\"FR\"","goff");

   // Now draw both histograms side-by-side ("same" option) as vertical bar charts ("bar2" option)
   hDiv->SetBarWidth(0.45);
   hDiv->SetBarOffset(0.1);
   hDiv->SetFillColor(49);
   TH1 *h1 = hDiv->DrawCopy("bar2");
   hDivFR->SetBarWidth(0.4);
   hDivFR->SetBarOffset(0.55);
   hDivFR->SetFillColor(50);
   TH1 *h2 = hDivFR->DrawCopy("bar2,same");

   TLegend *legend = new TLegend(0.55,0.65,0.76,0.82);
   legend->AddEntry(h1,"All nations","f");
   legend->AddEntry(h2,"French only","f");
   legend->Draw();

   c1->cd();

   return c1;
}
