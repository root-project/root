void P020_TQt6GuiFactory()
{
   gPluginMgr->AddHandler("TGuiFactory", "qt6", "ROOT::Experimental::TQt6GuiFactory",
      "ROOTQt6Canvas", "TQt6GuiFactory()");
}
