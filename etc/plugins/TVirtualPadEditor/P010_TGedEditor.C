void P010_TGedEditor()
{
   gPluginMgr->AddHandler("TVirtualPadEditor", "Ged", "TGedEditor",
      "Ged", "TGedEditor(TCanvas*)");
}
