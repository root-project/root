void P010_TPad()
{
   gPluginMgr->AddHandler("TVirtualPad", "*", "TPad",
      "Gpad", "TPad()");
}
