void P010_THistPainter()
{
   gPluginMgr->AddHandler("TVirtualHistPainter", "*", "THistPainter",
      "HistPainter", "THistPainter()");
}
