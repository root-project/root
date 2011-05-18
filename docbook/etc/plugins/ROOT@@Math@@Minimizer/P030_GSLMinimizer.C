void P030_GSLMinimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "GSLMultiMin", "ROOT::Math::GSLMinimizer",
      "MathMore", "GSLMinimizer(const char *)");
}
