void P020_TGLSAViewer()
{
   gPluginMgr->AddHandler("TVirtualViewer3D", "ogl", "TGLSAViewer",
      "RGL", "TGLSAViewer(TVirtualPad*)");
}
