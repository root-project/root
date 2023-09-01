void P060_TLinearMinimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Linear", "TLinearMinimizer",
      "Minuit", "TLinearMinimizer(const char *)");
}
