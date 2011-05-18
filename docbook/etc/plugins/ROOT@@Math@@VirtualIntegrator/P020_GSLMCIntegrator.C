void P020_GSLMCIntegrator()
{
   gPluginMgr->AddHandler("ROOT::Math::VirtualIntegrator", "GSLMCIntegrator", "ROOT::Math::GSLMCIntegrator",
      "MathMore", "GSLMCIntegrator(const char *, double, double, unsigned int)");
}
