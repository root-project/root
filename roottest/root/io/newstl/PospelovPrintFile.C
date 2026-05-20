void PospelovPrintFile(const char * fname = "pospelov.2010.mc10_7TeV.pool.root") 
{
  TFile * f = new TFile(fname);
#ifdef ClingWorkAroundMissingDynamicScope
  TTree *CollectionTree; f->GetObject("CollectionTree",CollectionTree);
#endif
  TBranch * cbr = CollectionTree->GetBranch("CaloLocalHadCoeff_EMFracClassify");
  gDebug = 0;
  cbr->GetEntry(0);
  gDebug = 0;
  CaloLocalHadCoeff * m_data = *((CaloLocalHadCoeff **)cbr->GetAddress());
  std::cout << "Data has " << m_data->m_AreaSet.size(); // getSizeAreaSet() 
  std::cout << " areas defined" << std::endl;
  for (int iA=0;iA < m_data->m_AreaSet.size(); iA++ ) {
    std::cout << " Area " << iA << " has title " 
	      << m_data->m_AreaSet.at(iA).m_title 
	      << " and " 
	      << m_data->m_AreaSet.at(iA).m_dims.size() 
	      << " dimensions" << std::endl;
    for (int iD=0;iD<m_data->m_AreaSet.at(iA).m_dims.size(); iD++ ) {
      std::cout << "  Dim " << iD << " has title " 
		<< m_data->m_AreaSet.at(iA).m_dims.at(iD).m_title 
		<< " and " 
		<< m_data->m_AreaSet.at(iA).m_dims.at(iD).m_nbins 
		<< " bins from " 
		<< m_data->m_AreaSet.at(iA).m_dims.at(iD).m_xmin 
		<< " to " 
		<< m_data->m_AreaSet.at(iA).m_dims.at(iD).m_xmax
		<< std::endl;
    }
  }
}
