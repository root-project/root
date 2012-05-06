// RooFit headers
#include "RooWorkspace.h"
#include "RooRealVar.h"
// RooStats headers
#include "RooStats/ModelConfig.h"

using namespace RooStats;

//__________________________________________________________________________________
void buildPoissonProductModel(RooWorkspace *w)
{
   // Build product model
   w->factory("expr::compsig('2*sig*pow(1.2, beta)', sig[0,20], beta[-5,5])");
   w->factory("Poisson::poiss1(x[0,40], sum::splusb1(sig, bkg1[0,20]))");
   w->factory("Poisson::poiss2(y[0,120], sum::splusb2(compsig, bkg2[0,20]))");
   w->factory("Poisson::constr1(gbkg1[10,0,20], bkg1)");
   w->factory("Poisson::constr2(gbkg2[10,0,20], bkg2)");
   w->factory("Gaussian::constr3(beta0[0,-5,5], beta, 1)"); 
   w->factory("PROD::pdf(poiss1, poiss2, constr1, constr2, constr3)");

   // set POI prior Pdf (for BayesianCalculator and other Bayesian methods) 
   w->factory("Uniform::prior(sig)");

   // build argument sets
   w->defineSet("obs", "x,y");
   w->defineSet("poi", "sig");
   w->defineSet("nuis", "bkg1,bkg2,beta");
   w->defineSet("globObs", "beta0,gbkg1,gbkg2");

   // set global observables to constant values
   RooFIter iter = w->set("globObs")->fwdIterator();
   RooRealVar *var;
   while((var = (RooRealVar *)iter.next()) != NULL) var->setConstant();

   // build data set and import it into the workspace sets
   RooDataSet *data = new RooDataSet("data", "data", *w->set("obs"));
   w->import(*data);

   // create signal + background model configuration
   ModelConfig *sbModel = new ModelConfig("S+B", w);
   sbModel->SetObservables(*w->set("obs"));
   sbModel->SetGlobalObservables(*w->set("globObs"));
   sbModel->SetParametersOfInterest(*w->set("poi"));
   sbModel->SetNuisanceParameters(*w->set("nuis"));
   sbModel->SetPdf("pdf");
   sbModel->SetPriorPdf("prior");

   // create background model configuration
   ModelConfig *bModel = new ModelConfig(*sbModel);
   bModel->SetName("B");
   w->var("sig")->setVal(0);
   bModel->SetSnapshot(*w->set("poi"));

   w->import(*sbModel);
   w->import(*bModel);
}


void buildOnOffModel(RooWorkspace *w)
{
   // Build model for prototype on/off problem
   // Poiss(x | s+b) * Poiss(y | tau b )
   w->factory("Poisson::on_pdf(n_on[0,500],sum::splusb(sig[0,500],bkg[0,500]))");
   w->factory("Poisson::off_pdf(n_off[0,500],prod::taub(tau[0.1,5.0],bkg))");
   w->factory("PROD::prod_pdf(on_pdf, off_pdf)");        

   // construct the Bayesian-averaged model (eg. a projection pdf)
   // p'(x|s) = \int db p(x|s+b) * [ p(y|b) * prior(b) ]
   w->factory("Uniform::prior(bkg)");
   w->factory("PROJ::averagedModel(PROD::foo(on_pdf|bkg,off_pdf,prior),bkg)") ;

   // define sets of variables obs={x} and poi={sig}
   // x is the only observable in the main measurement and y is treated as a separate measurement,
   // which is used to produce the prior that will be used in the calculation to randomize the nuisance parameters
   w->defineSet("obs", "n_on,n_off,tau");
   w->defineSet("poi", "sig");
   w->defineSet("nuis", "bkg");

   // define data set and import it into workspace
   RooDataSet *data = new RooDataSet("data", "data", *w->set("obs"));
   w->import(*data);

   // create signal + background model configuration
   ModelConfig *sbModel = new ModelConfig("S+B", w);
   sbModel->SetPdf(*w->pdf("prod_pdf"));
   sbModel->SetObservables(*w->set("obs"));      
   sbModel->SetParametersOfInterest(*w->set("poi"));
   sbModel->SetNuisanceParameters(*w->set("nuis"));

   // create background model configuration
   ModelConfig *bModel = new ModelConfig(*sbModel);
   bModel->SetName("B");
   
   // alternate priors
   w->factory("Gaussian::gauss_prior(bkg, n_off, expr::sqrty('sqrt(n_off)', n_off))");
   w->factory("Lognormal::lognorm_prior(bkg, n_off, expr::kappa('1+1./sqrt(n_off)',n_off))");

   w->import(*sbModel);
   w->import(*bModel);
}

