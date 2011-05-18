void P010_Minuit2Minimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Minuit2", "ROOT::Minuit2::Minuit2Minimizer",
      "Minuit2", "Minuit2Minimizer(const char *)");
}
