void P020_TMinuitMinimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Minuit", "TMinuitMinimizer",
      "Minuit", "TMinuitMinimizer(const char *)");
}
