void P020_TQt6GedEditor()
{
   gPluginMgr->AddHandler("TVirtualPadEditor", "qt6", "ROOT::Experimental::TQt6GedEditor",
      "ROOTQt6Canvas", "TQt6GedEditor(TCanvas*)");
}
