void P010_TGraphPainter()
{
   gPluginMgr->AddHandler("TVirtualGraphPainter", "*", "TGraphPainter",
      "HistPainter", "TGraphPainter()");
}
