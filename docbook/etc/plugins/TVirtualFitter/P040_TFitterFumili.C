void P040_TFitterFumili()
{
   gPluginMgr->AddHandler("TVirtualFitter", "Fumili2", "TFitterFumili",
      "Minuit2", "TFitterFumili(Int_t)");
}
