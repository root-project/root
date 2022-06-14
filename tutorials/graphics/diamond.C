/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Draw a diamond.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

TCanvas *diamond(){
   TCanvas *c = new TCanvas("c");
   TDiamond *d = new TDiamond(.05,.1,.95,.8);

   d->AddText("A TDiamond can contain any text.");

   d->Draw();
   return c;
}
