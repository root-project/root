//Draw a diamond
//Author: Olivier Couet
TCanvas *diamond(){
   TCanvas *c = new TCanvas("c");
   TDiamond *d = new TDiamond(.05,.1,.95,.8);

   d->AddText("A TDiamond can contain any text.");

   d->Draw();
   return c;
}
