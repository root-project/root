void P020_Bisection()
{
   gPluginMgr->AddHandler("ROOT::Math::IRootFinderMethod", "Bisection", "ROOT::Math::Roots::Bisection",
                          "MathMore", "Bisection()");
}
