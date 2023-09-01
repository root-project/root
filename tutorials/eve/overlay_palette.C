/// \file
/// \ingroup tutorial_eve
///
/// \macro_code
///
/// \author Matevz Tadel

void overlay_palette()
{
   TEveManager::Create();
   auto p  = new TEveRGBAPalette(0, 100);
   auto po = new TEveRGBAPaletteOverlay(p, 0.55, 0.1, 0.4, 0.05);
   auto v  = gEve->GetDefaultGLViewer();
   v->AddOverlayElement(po);
}
