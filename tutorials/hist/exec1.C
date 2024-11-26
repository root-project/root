/// \file
/// \ingroup tutorial_hist
/// Echo object at mouse position.
/// Example of macro called when a pad is redrawn
/// one must create a TExec object in the following way
/// ~~~{.cpp}
///    gPad->AddExec("ex1", ".x exec1.C");
/// ~~~
/// this macro prints the bin number and the bin content when one clicks
/// on the histogram contour of any histogram in a pad
///
/// \macro_code
///
/// \authors Rene Brun, Sergey Linev


void exec1()
{
   if (!gPad) {
      Error("exec1", "gPad is null, you are not supposed to run this macro");
      return;
   }

   Int_t event = gPad->GetEvent();
   int px = gPad->GetEventX();
   TObject *select = gPad->GetSelected();

   if (select && select->InheritsFrom(TH1::Class())) {
      TH1 *h = (TH1*)select;
      Float_t xx = gPad->AbsPixeltoX(px);
      Float_t x  = gPad->PadtoX(xx);
      Int_t binx = h->GetXaxis()->FindBin(x);
      printf("event=%d, hist:%s, bin=%d, content=%f\n", event, h->GetName(), binx, h->GetBinContent(binx));
   }
}

