void P010_TVirtualViewerX3D()
{
   gPluginMgr->AddHandler("TVirtualViewer3D", "x3d", "TViewerX3D",
      "X3d", "TViewerX3D(TVirtualPad*)");
}
