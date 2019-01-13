void P100_Ipopt()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Ipopt", "ROOT::Math::Experimental::IpoptMinimizer", "Ipopt",
                          "IpoptMinimizer(const char*)");
}
