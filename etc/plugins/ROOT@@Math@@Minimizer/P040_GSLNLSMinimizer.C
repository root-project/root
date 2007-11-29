void P040_GSLNLSMinimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "GSLMultiFit", "ROOT::Math::GSLNLSMinimizer",
      "MathMore", "GSLNLSMinimizer(int)");
}
