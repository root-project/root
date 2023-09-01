void P040_Newton()
{
   gPluginMgr->AddHandler("ROOT::Math::IRootFinderMethod", "Newton", "ROOT::Math::Roots::Newton",
                          "MathMore", "Newton()");
}
