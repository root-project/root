/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Draw a diamond.
///
/// \macro_image (tcanvas_js)
/// \preview 
/// \macro_code
///
/// \author Olivier Couet
/// \date February 2024

void diamond(){
   auto d = new TDiamond(.05,.1,.95,.8);
   d->AddText("A TDiamond can contain any text.");
   d->Draw();
}
