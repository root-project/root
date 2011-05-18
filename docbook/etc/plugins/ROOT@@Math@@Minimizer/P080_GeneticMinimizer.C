void P080_GeneticMinimizer()
{
   gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Genetic", "ROOT::Math::GeneticMinimizer",
      "Genetic", "GeneticMinimizer(int)");
}
