void P010_TFitter()
{
   gPluginMgr->AddHandler("TVirtualFitter", "Minuit", "TFitter",
      "Minuit", "TFitter(Int_t)");
}
