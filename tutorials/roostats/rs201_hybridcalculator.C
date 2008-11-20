void rs201_hybridcalculator()
{
  /// This macro show an example on how to use the RooStats/HybridCalculator class
 
  //***********************************************************************//

  using namespace RooFit;
  using namespace RooStats;

  /// set random seed
  // to do

  /// build the models for background and signal+background
  RooRealVar x("x","",-3,3);
  RooArgList observables(x); // variables to be generated

  // gaussian signal
  RooRealVar sig_mean("sig_mean","",0);
  RooRealVar sig_sigma("sig_sigma","",0.8);
  RooGaussian sig_pdf("sig_pdf","",x,sig_mean,sig_sigma);
  RooRealVar sig_yield("sig_yield","",100,0,300);

  // flat background (extended PDF)
  RooRealVar bkg_slope("bkg_slope","",0);
  RooPolynomial bkg_pdf("bkg_pdf","",x,bkg_slope);
  RooRealVar bkg_yield("bkg_yield","",100,0,300);
  RooExtendPdf bkg_ext_pdf("bkg_ext_pdf","",bkg_pdf,bkg_yield);

  // total sig+bkg (extended PDF)
  RooAddPdf tot_pdf("tot_pdf","",RooArgList(sig_pdf,bkg_pdf),RooArgList(sig_yield,bkg_yield));

  /// build the prior PDF on the parameters to be integrated
  // gaussian contraint on the background yield ( N_B = 100 +/- 10 )
  RooGaussian bkg_yield_prior("bkg_yield_prior","",bkg_yield,RooConst(100.),RooConst(10.));
  RooArgSet nuisance_parameters(bkg_yield); // variables to be integrated

  /// generate a data sample
  RooDataSet* data = tot_pdf.generate(observables,RooFit::Extended());

  //***********************************************************************//

  /// run HybridCalculator on those inputs
  HybridCalculator myHybridCalc("myHybridCalc","HybridCalculator example",tot_pdf,bkg_ext_pdf,observables,nuisance_parameters,bkg_yield_prior);

  // here I use the default test statistics: 2*lnQ (optional)
  myHybridCalc->SetTestStatistics(1);

  // run 1000 toys with gaussian prior on the background yield
  HybridResult* myHybridResult = myHybridCalc->Calculate(*data,1000,true);

  // run 1000 toys without gaussian prior on the background yield
  //HybridResult* myHybridResult = myHybridCalc->Calculate(*data,1000,false);

  /// save the toy-MC results to file, this way splitting into sub-batch jobs is possible
  //myHybridResult->SaveToFile("hybridresult_results.root");

  /// example on how to merge with toy-MC results obtained in another job
  //HybridResult* mergedHybridResult = new HybridResult("mergedHybridResult","this object holds merged results");
  //mergedHybridResult->Add(myHybridResult);
  //mergedHybridResult->AddFromFile("other_hybridresult_results.root");

  /// recover and display the results
  double clsb_data = myHybridResult->CLsplusb();
  double clb_data = myHybridResult->CLb();
  double cls_data = myHybridResult->CLs();

  /// to do: more results on mean CL_sb, RMS and mean of the test statistics, ...

  HybridPlot* myPlot = myHybridResult->GetPlot();

  std::cout << "Completed HybridCalculator example:\n"; 
  std::cout << " - CL_sb = " << clsb_data << std::endl;
  std::cout << " - CL_b = " << clb_data << std::endl;
  std::cout << " - CL_s = " << cls_data << std::endl;

  // for this example, you should get: CL_sb =  and CL_b = 
}