void P010_GSLIntegrator()
{
   gPluginMgr->AddHandler("ROOT::Math::VirtualIntegrator", "GSLIntegrator", "ROOT::Math::GSLIntegrator",
      "MathMore", "GSLIntegrator(const char *, int, double, double, unsigned int)");
}
