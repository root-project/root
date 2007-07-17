void P020_TQtViewerX3D()
{
   gPluginMgr->AddHandler("TViewerX3D", "qt", "TQtViewerX3D",
      "QtX3d", "TQtViewerX3D(TVirtualPad*,Option_t*,const char*,UInt_t,UInt_t)");
}
