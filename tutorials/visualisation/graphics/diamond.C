/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// \preview Draw a diamond.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void diamond()
{
   auto d = new TDiamond(.05, .1, .95, .8);
   d->AddText("A TDiamond can contain any text.");
   d->Draw();
}
