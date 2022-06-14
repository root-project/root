void P010_TGLHistPainter()
{
   gPluginMgr->AddHandler("TGLHistPainter", "*", "TGLHistPainter",
      "RGL", "TGLHistPainter(TH1*)");
}
