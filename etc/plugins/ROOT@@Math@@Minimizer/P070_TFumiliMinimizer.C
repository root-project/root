void P070_TFumiliMinimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Fumili", "TFumiliMinimizer",
      "Fumili", "TFumiliMinimizer(int)");
}
