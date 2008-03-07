void P050_Secant()
{
   gPluginMgr->AddHandler("ROOT::Math::IRootFinderMethod", "Secant", "ROOT::Math::Roots::Secant",
                          "MathMore", "Secant()");
}
