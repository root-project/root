void P030_FalsePos()
{
   gPluginMgr->AddHandler("ROOT::Math::IRootFinderMethod", "FalsePos", "ROOT::Math::Roots::FalsePos",
                          "MathMore", "FalsePos()");
}
