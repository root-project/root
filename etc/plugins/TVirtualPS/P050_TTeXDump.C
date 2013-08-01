void P050_TTeXDump()
{
   gPluginMgr->AddHandler("TVirtualPS", "tex", "TTeXDump",
      "Postscript", "TTeXDump()");
}
