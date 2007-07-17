void P010_TPostScript()
{
   gPluginMgr->AddHandler("TVirtualPS", "ps", "TPostScript",
      "Postscript", "TPostScript()");
}
