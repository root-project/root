void P010_Brent()
{
   gPluginMgr->AddHandler("ROOT::Math::IRootFinderMethod", "Brent", "ROOT::Math::Roots::Brent",
                          "MathMore", "Brent()");
}
