void P090_IpoptMinimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Ipopt", "ROOT::Math::IpoptMinimizer",
      "Ipopt", "IpoptMinimizer(const char*)");
}
