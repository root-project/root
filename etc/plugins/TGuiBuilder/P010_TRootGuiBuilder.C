void P010_TRootGuiBuilder()
{
   gPluginMgr->AddHandler("TGuiBuilder", "*", "TRootGuiBuilder",
      "GuiBld", "TRootGuiBuilder()");
}
