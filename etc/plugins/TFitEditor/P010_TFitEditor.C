void P010_TFitEditor()
{
   gPluginMgr->AddHandler("TFitEditor", "*", "TFitEditor",
      "FitPanel", "GetInstance(TVirtualPad*, TObject*)");
}
