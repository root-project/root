void P010_GSLIntegrator()
{
   gPluginMgr->AddHandler("ROOT::Math::VirtualIntegrator", "GSLIntegrator", "ROOT::Math::GSLIntegrator",
      "GSLIntegrator", "GSLIntegrator(const char *, double, double, unsigned int)");
}
