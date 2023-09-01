void P020_TSVG()
{
   gPluginMgr->AddHandler("TVirtualPS", "svg", "TSVG",
      "Postscript", "TSVG()");
}
