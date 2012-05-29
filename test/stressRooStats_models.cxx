// RooFit headers
#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooDataSet.h"

// RooStats headers
#include "RooStats/ModelConfig.h"

using namespace RooFit;
using namespace RooStats;

//__________________________________________________________________________________
void buildPoissonProductModel(RooWorkspace *w)
{
   // Build product model
   w->factory("expr::compsig('2*sig*pow(1.2, beta)', sig[0,20], beta[-3,3])");
   w->factory("Poisson::poiss1(x[0,40], sum::splusb1(sig, bkg1[0,10]))");
   w->factory("Poisson::poiss2(y[0,120], sum::splusb2(compsig, bkg2[0,10]))");
   w->factory("Poisson::constr1(gbkg1[5,0,10], bkg1)");
   w->factory("Poisson::constr2(gbkg2[5,0,10], bkg2)");
   w->factory("Gaussian::constr3(beta0[0,-3,3], beta, 1)");
   w->factory("PROD::pdf(poiss1, poiss2, constr1, constr2, constr3)");

   // POI prior Pdf (for BayesianCalculator and other Bayesian methods)
   w->factory("Uniform::prior(sig)");
   // Nuisance parameters Pdf (for HybridCalculator)
   w->factory("PROD::prior_nuis(constr1,constr2,constr3)");

   // extended pdfs and simultaneous pdf
   w->factory("ExtendPdf::epdf1(PROD::pdf1(poiss1,constr1),n1[0,10])");
   w->factory("ExtendPdf::epdf2(PROD::pdf2(poiss2,constr2,constr3),n2[0,10])");
   w->factory("SIMUL::sim_pdf(index[cat1,cat2],cat1=epdf1,cat2=epdf2)");

   // build argument sets
//   w->defineSet("obs", "x,y");
//   w->defineSet("poi", "sig");
//   w->defineSet("nuis", "bkg1,bkg2,beta");
//   w->defineSet("globObs", "beta0,gbkg1,gbkg2");

   // create signal + background model configuration
   ModelConfig *sbModel = new ModelConfig("S+B", w);
   sbModel->SetObservables("x,y");
   sbModel->SetGlobalObservables("beta0,gbkg1,gbkg2");
   sbModel->SetParametersOfInterest("sig");
   sbModel->SetNuisanceParameters("bkg1,bkg2,beta");
   sbModel->SetPdf("pdf");

   // create background model configuration
   ModelConfig *bModel = new ModelConfig(*sbModel);
   bModel->SetName("B");

   // set global observables to constant values
   RooFIter iter = sbModel->GetGlobalObservables()->fwdIterator();
   RooRealVar *var = dynamic_cast<RooRealVar *>(iter.next());
   while(var != NULL) {
      var->setConstant(); 
      var = dynamic_cast<RooRealVar *>(iter.next());
   }

   // build data set and import it into the workspace sets
   RooDataSet *data = new RooDataSet("data", "data", *sbModel->GetObservables());
   w->import(*data);

   w->import(*sbModel);
   w->import(*bModel);
}


//__________________________________________________________________________________
// Insightful comments on model courtesy of Kyle Cranmer, Wouter Verkerke, Sven Kreiss
//    from $ROOTSYS/tutorials/roostats/HybridInstructional.C
void buildOnOffModel(RooWorkspace *w)
{
   // Build model for prototype on/off problem
   // Poiss(x | s+b) * Poiss(y | tau b )
   w->factory("Poisson::on_pdf(n_on[0,300],sum::splusb(sig[0,100],bkg[0,200]))");
   w->factory("Poisson::off_pdf(n_off[0,1100],prod::taub(tau[0.1,5.0],bkg))");
   w->factory("PROD::prod_pdf(on_pdf, off_pdf)");

   // construct the Bayesian-averaged model (eg. a projection pdf)
   // p'(x|s) = \int db p(x|s+b) * [ p(y|b) * prior(b) ]
   w->factory("Uniform::prior(bkg)");
   w->factory("PROJ::averagedModel(PROD::foo(on_pdf|bkg,off_pdf,prior),bkg)") ;

   // create signal + background model configuration
   ModelConfig *sbModel = new ModelConfig("S+B", w);
   sbModel->SetPdf("prod_pdf");
   sbModel->SetObservables("n_on,n_off");
   sbModel->SetParametersOfInterest("sig");
   sbModel->SetNuisanceParameters("bkg");

   // create background model configuration
   ModelConfig *bModel = new ModelConfig(*sbModel);
   bModel->SetName("B");

   // alternate priors
   w->factory("Gaussian::gauss_prior(bkg, n_off, expr::sqrty('sqrt(n_off)', n_off))");
   w->factory("Lognormal::lognorm_prior(bkg, n_off, expr::kappa('1+1./sqrt(n_off)',n_off))");

   // define data set and import it into workspace
   RooDataSet *data = new RooDataSet("data", "data", *sbModel->GetObservables());
   w->import(*data);

   w->import(*sbModel);
   w->import(*bModel);
}


void createPoissonEfficiencyModel(RooWorkspace *w)
{

   // build models
   w->factory("Gaussian::constrb(b0[-5,5], b1[-5,5], 1)");
   w->factory("Gaussian::constre(e0[-5,5], e1[-5,5], 1)");
   w->factory("expr::bkg('5 * pow(1.3, b1)', b1)"); // background model
   w->factory("expr::eff('0.5 * pow(1.2, e1)', e1)"); // efficiency model
   w->factory("expr::splusb('eff * sig + bkg', eff, bkg, sig[0,20])");
   w->factory("Poisson::sb_poiss(x[0,40], splusb)");
   w->factory("Poisson::b_poiss(x, bkg)");
   w->factory("PROD::sb_pdf(sb_poiss, constrb, constre)");
   w->factory("PROD::b_pdf(b_poiss, constrb)");
   w->factory("PROD::priorbkg(constr1, constr2)");

   w->var("b0")->setConstant(kTRUE);
   w->var("e0")->setConstant(kTRUE);

   // build argument sets
   w->defineSet("obs", "x");
   w->defineSet("poi", "sig");
   w->defineSet("nuis", "b1,e1");
   w->defineSet("globObs", "b0,e0");

   // define data set and import it into workspace
   RooDataSet *data = new RooDataSet("data", "data", *w->set("obs"));
   w->import(*data);

   // create model configuration
   ModelConfig *sbModel = new ModelConfig("S+B", w);
   sbModel->SetObservables("x");
   sbModel->SetParametersOfInterest("sig");
   sbModel->SetNuisanceParameters("b1,e1");
   sbModel->SetGlobalObservables("b0,e0");
   sbModel->SetPdf("sb_pdf");
   //sbModel->SetPriorPdf("prior");
   sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());

   ModelConfig *bModel = new ModelConfig(*sbModel);
   bModel->SetName("B");
   bModel->SetPdf("b_pdf");

   w->import(*sbModel);
   w->import(*bModel);
}


