// echo object at mouse position
void exec1()
{
   //example of macro called when a pad is redrawn
   //one must create a TExec object in the following way
   // TExec ex("ex",".x exec1.C");
   // ex.Draw();
   // this macro prints the bin number and the bin content when one clicks
   //on the histogram contour of any histogram in a pad
   //Author: Rene Brun

   if (!gPad) {
      Error("exec1", "gPad is null, you are not supposed to run this macro");
      return;
   }

   int event = gPad->GetEvent();
   if (event != 11) return;
   int px = gPad->GetEventX();
   TObject *select = gPad->GetSelected();
   if (!select) return;
   if (select->InheritsFrom(TH1::Class())) {
      TH1 *h = (TH1*)select;
      Float_t xx = gPad->AbsPixeltoX(px);
      Float_t x  = gPad->PadtoX(xx);
      Int_t binx = h->GetXaxis()->FindBin(x);
      printf("event=%d, hist:%s, bin=%d, content=%f\n",event,h->GetName(),binx,h->GetBinContent(binx));
   }
}

