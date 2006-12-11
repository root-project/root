//Fitting with the RooFit package
//Author: Wouter Verkerke
   
void RoofitDemo()
{
  gSystem->Load("libRooFit") ;
  using namespace RooFit ;

  // --- Build Gaussian PDFs ---
  RooRealVar mes("mes","m_{ES} (GeV)",5.20,5.30) ;
  RooRealVar sigmean("sigmean","B^{#pm} mass",5.28,5.20,5.30) ;
  RooRealVar sigwidth("sigwidth","B^{#pm} width",0.0027,0.001,1.) ;
  RooGaussian gauss("gauss","gaussian PDF",mes,sigmean,sigwidth) ;  
  
  // --- Build Argus background PDF ---
  RooRealVar argpar("argpar","argus shape parameter",-20.0,-100.,-1.) ;
  RooRealVar cutoff("cutoff","argus cutoff",5.291) ;
  RooArgusBG argus("argus","Argus PDF",mes,cutoff,argpar) ;

  // --- Construct composite PDF ---
  RooRealVar nsig("nsig","number of signal events",200,0.,10000) ;
  RooRealVar nbkg("nbkg","number of background events",800,0.,10000) ;
  RooAddPdf sum("sum","gauss+argus",RooArgList(gauss,argus),RooArgList(nsig,nbkg)) ;

  // --- Generate a toyMC sample from composite PDF ---
  RooDataSet *data = sum.generate(mes,2000) ;

  // --- Perform extended ML fit of composite PDF to toy data, skip minos ---
  RooFitResult* r = sum.fitTo(*data,Extended(),Minos(kFALSE),Save(kTRUE)) ;
  
  // --- Plot toy data and composite PDF overlaid ---
  RooPlot* mesframe = mes.frame(Name("mesframe"),Title("B^{#pm} #rightarrow D^{0}K^{#pm}")) ;
  data->plotOn(mesframe) ;
  sum.plotOn(mesframe) ;

  // --- Overlay background-only component of composite PDF with dashed line style ---
  sum.plotOn(mesframe,Components(argus),LineStyle(kDashed)) ;

  // --- Add box with subset of fit parameters to plot frame ---
  nsig.setPlotLabel("N(B^{#pm})") ;
  sigmean.setPlotLabel("B^{#pm} mass") ;
  sigwidth.setPlotLabel("B^{#pm} width") ;
  sum.paramOn(mesframe,Parameters(RooArgSet(nsig,sigmean,sigwidth)),Layout(0.15,0.55,0.85)) ;

  // --- Plot frame on canvas ---
  mesframe->Draw() ;

  // --- Print dump of fit results ---
  r->Print("v") ;

  // --- Delete generate toy data ---
  delete data ;
}

