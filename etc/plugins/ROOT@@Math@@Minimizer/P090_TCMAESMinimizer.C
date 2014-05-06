void P090_TCMAESMinimizer()
{
  gPluginMgr->AddHandler("ROOT::Math::Minimizer", "cmaes", "ROOT::cmaes::TCMAESMinimizer",
			 "cmaesroot", "TCMAESMinimizer(const char *)");
}
