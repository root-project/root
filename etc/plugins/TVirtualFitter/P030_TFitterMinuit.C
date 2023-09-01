void P030_TFitterMinuit()
{
   gPluginMgr->AddHandler("TVirtualFitter", "Minuit2", "TFitterMinuit",
      "Minuit2", "TFitterMinuit(Int_t)");
}
