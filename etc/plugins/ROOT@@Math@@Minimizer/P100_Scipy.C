void P100_Scipy()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Scipy", "ROOT::Math::Experimental::ScipyMinimizer", "Scipy",
                          "ScipyMinimizer(const char*)");
}
