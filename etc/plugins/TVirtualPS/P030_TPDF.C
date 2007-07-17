void P030_TPDF()
{
   gPluginMgr->AddHandler("TVirtualPS", "pdf", "TPDF",
      "Postscript", "TPDF()");
}
