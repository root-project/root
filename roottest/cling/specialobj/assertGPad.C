// check gPad access
void assertGPad() {
   new TCanvas();
   for (int i=0;i<1;++i) {
      if (i) gPad->SetLogy();
   }
}
