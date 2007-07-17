void P010_TViewerX3D()
{
   gPluginMgr->AddHandler("TViewerX3D", "x11", "TViewerX3D",
      "X3d", "TViewerX3D(TVirtualPad*,Option_t*,const char*,UInt_t,UInt_t)");
}
