void P090_TCMAESMinimizer()
{
  gPluginMgr->AddHandler("ROOT::Math::Minimizer", "cmaes", "ROOT::cmaes::TCMAESMinimizer",
			 "cmaes", "TCMAESMinimizer(const char *)");
}
