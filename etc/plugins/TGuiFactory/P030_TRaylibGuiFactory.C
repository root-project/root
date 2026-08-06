void P030_TRaylibGuiFactory()
{
   gPluginMgr->AddHandler("TGuiFactory", "raylib", "ROOT::Experimental::TRaylibGuiFactory",
      "ROOTRaylibCanvas", "TRaylibGuiFactory()");
}
