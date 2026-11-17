void P030_TGtk4GedEditor()
{
   gPluginMgr->AddHandler("TVirtualPadEditor", "gtk4", "ROOT::Experimental::TGtk4GedEditor",
      "ROOTGtk4Canvas", "TGtk4GedEditor(TCanvas*)");
}
