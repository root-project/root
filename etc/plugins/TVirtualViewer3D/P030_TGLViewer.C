void P030_TGLViewer()
{
   gPluginMgr->AddHandler("TVirtualViewer3D", "gl", "TGLViewer",
      "RGL", "TGLViewer(TVirtualPad*)");
}
