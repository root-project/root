void P010_TGLPadPainter()
{
   gPluginMgr->AddHandler("TVirtualPadPainter", "gl", "TGLPadPainter",
                          "RGL", "TGLPadPainter()");
}
