void P090_TCMAESMinimizer()
{
  gPluginMgr->AddHandler("ROOT::Math::Minimizer", "cmaes", "ROOT::cmaes::TCMAESMinimizer",
			 "cmaes_root", "TCMAESMinimizer(const char *)");
}
