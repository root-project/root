void P020_TFoamSampler()
{
   gPluginMgr->AddHandler("ROOT::Math::DistSampler", "Foam", "TFoamSampler",
      "Foam", "TFoamSampler( )");
}
