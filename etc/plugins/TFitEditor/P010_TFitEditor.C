void P010_TFitEditor()
{
   gPluginMgr->AddHandler("TFitEditor", "*", "TFitEditor",
      "FitPanel", "TFitEditor(const TVirtualPad*, const TObject*)");
}
