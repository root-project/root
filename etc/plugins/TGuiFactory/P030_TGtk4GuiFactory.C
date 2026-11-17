void P030_TGtk4GuiFactory()
{
   gPluginMgr->AddHandler("TGuiFactory", "gtk4", "ROOT::Experimental::TGtk4GuiFactory",
      "ROOTGtk4Canvas", "TGtk4GuiFactory()");
}
