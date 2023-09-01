void P050_GSLSimAnMinimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "GSLSimAn", "ROOT::Math::GSLSimAnMinimizer",
      "MathMore", "GSLSimAnMinimizer(int)");
}
