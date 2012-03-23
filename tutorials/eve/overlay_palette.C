void overlay_palette()
{ 
   TEveManager::Create();

   TEveRGBAPalette *p = new TEveRGBAPalette(0, 100);
   TEveRGBAPaletteOverlay *po = new TEveRGBAPaletteOverlay(p, 0.55, 0.1, 0.4, 0.05);

   TGLViewer* v = gEve->GetDefaultGLViewer();
   v->AddOverlayElement(po);
}
