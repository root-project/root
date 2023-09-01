void P060_Steffenson()
{
   gPluginMgr->AddHandler("ROOT::Math::IRootFinderMethod", "Steffenson", "ROOT::Math::Roots::Steffenson",
      "MathMore", "Steffenson()");
}
