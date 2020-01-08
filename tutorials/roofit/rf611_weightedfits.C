/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Likelihood and minimization: Parameter uncertainties for weighted unbinned ML fits
///
/// \macro_image
/// \macro_output
/// \macro_code
/// \author 11/2019 - Christoph Langenbruch

#include "TH1D.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TRandom3.h"
#include "TLegend.h"
#include "RooRealVar.h"
#include "RooFitResult.h"
#include "RooDataSet.h"
#include "RooPolynomial.h"

using namespace RooFit;


int rf611_weightedfits(int acceptancemodel=2) {
  // P a r a m e t e r   u n c e r t a i n t i e s   f o r   w e i g h t e d   u n b i n n e d   M L   f i t s
  // ---------------------------------------------------------------------------------------------------------
  //
  //Based on example from https://arxiv.org/abs/1911.01303 
  //
  //This example compares different approaches to determining parameter uncertainties in weighted unbinned maximum likelihood fits.
  //Performing a weighted unbinned maximum likelihood fits can be useful to account for acceptance effects and to statistically subtract background events using the sPlot formalism. 
  //It is however well known that the inverse Hessian matrix does not yield parameter uncertainties with correct coverage in the presence of event weights.
  //Three approaches to the determination of parameter uncertainties are compared in this example:
  //
  //1. Using the inverse weighted Hessian matrix [SumW2Error(false)]
  //
  //2. Using the expression [SumW2Error(true)]
  //   V_{ij} = H_{ik}^{-1} C_{kl} H_{lj}^{-1}
  //   where H is the weighted Hessian matrix and C is the Hessian matrix with squared weights
  //
  //3. The asymptotically correct approach (for details please see https://arxiv.org/abs/1911.01303) [Asymptotic(true)]
  //   V_{ij} = H_{ik}^{-1} D_{kl} H_{lj}^{-1}
  //   where H is the weighted Hessian matrix and D is given by
  //   D_{kl} = sum_{e=1}^{N} w_e^2 \frac{\partial log P}{\partial lambda_k}\frac{\partial log P}{\partial lambda_l}
  //   with the event weight w_e
  //
  //The example performs the fit of a second order polynomial in the angle cos(theta) [-1,1] to a weighted data set.
  //The polynomial is given by
  //  P = (1 + c0*cos(theta) + c1*cos(theta)*cos(theta)) / Norm
  //The two coefficients c0 and c1 and their uncertainties are to be determined in the fit.
  //
  //The per-event weight is used to correct for an acceptance effect, two different acceptance models can be studied:
  //  acceptancemodel==1: eff = 0.3+0.7*cos(theta)*cos(theta)
  //  acceptancemodel==2: eff = 1.0-0.7*cos(theta)*cos(theta)
  //The data is generated to be flat before the acceptance effect.
  //
  //The performance of the different approaches to determine parameter uncertainties is compared using the pull distributions from a large number of pseudoexperiments.
  //The pull is defined as (lambda_i-\lambda_{gen})/\sigma(\lambda_i), where \lambda_i is the fitted parameter and \sigma(\lambda_i) its uncertainty for pseudoexperiment number i.
  //If the fit is unbiased and the parameter uncertainties are estimated correctly, the pull distribution should be a Gaussian centered around zero with a width of one.

  
  // I n i t i a l i s a t i o n   a n d   S e t u p
  //------------------------------------------------  

  //plotting options
  gStyle->SetPaintTextFormat(".1f");
  gStyle->SetEndErrorSize(6.0);
  gStyle->SetTitleSize(0.05, "XY");
  gStyle->SetLabelSize(0.05, "XY");
  gStyle->SetTitleOffset(0.9, "XY");
  gStyle->SetTextSize(0.05);
  gStyle->SetPadLeftMargin(0.125);
  gStyle->SetPadBottomMargin(0.125);
  gStyle->SetPadTopMargin(0.075);
  gStyle->SetPadRightMargin(0.075);
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.0);
  gStyle->SetHistLineWidth(2.0);
  gStyle->SetHistLineColor(1);

  //initialise TRandom3
  TRandom3* rnd = new TRandom3();
  rnd->SetSeed(191101303);

  //accepted events and events weighted to account for the acceptance
  TH1D* haccepted = new TH1D("haccepted", "Generated events;cos(#theta);#events", 40, -1.0, 1.0);
  TH1D* hweighted = new TH1D("hweighted", "Generated events;cos(#theta);#events", 40, -1.0, 1.0);
  //histograms holding pull distributions  
  //using the inverse Hessian matrix
  TH1D* hc0pull1 = new TH1D("hc0pull1", "Inverse weighted Hessian matrix [SumW2Error(false)];Pull (c_{0}^{fit}-c_{0}^{gen})/#sigma(c_{0});", 20, -5.0, 5.0);
  TH1D* hc1pull1 = new TH1D("hc1pull1", "Inverse weighted Hessian matrix [SumW2Error(false)];Pull (c_{1}^{fit}-c_{1}^{gen})/#sigma(c_{1});", 20, -5.0, 5.0);
  //using the correction with the Hessian matrix with squared weights
  TH1D* hc0pull2 = new TH1D("hc0pull2", "Hessian matrix with squared weights [SumW2Error(true)];Pull (c_{0}^{fit}-c_{0}^{gen})/#sigma(c_{0});", 20, -5.0, 5.0);
  TH1D* hc1pull2 = new TH1D("hc1pull2", "Hessian matrix with squared weights [SumW2Error(true)];Pull (c_{1}^{fit}-c_{1}^{gen})/#sigma(c_{1});", 20, -5.0, 5.0);
  //asymptotically correct approach
  TH1D* hc0pull3 = new TH1D("hc0pull3", "Asymptotically correct approach [Asymptotic(true)];Pull (c_{0}^{fit}-c_{0}^{gen})/#sigma(c_{0});", 20, -5.0, 5.0);
  TH1D* hc1pull3 = new TH1D("hc1pull3", "Asymptotically correct approach [Asymptotic(true)];Pull (c_{1}^{fit}-c_{1}^{gen})/#sigma(c_{1});", 20, -5.0, 5.0);

  //number of pseudoexperiments (toys) and number of events per pseudoexperiment
  unsigned int ntoys = 500;
  unsigned int nstats = 5000;
  //parameters used in the generation
  double c0gen = 0.0;
  double c1gen = 0.0;

  // M a i n   l o o p   o v e r   a l l   p s e u d o e x p e r i m e n t s
  //------------------------------------------------------------------------
  for (unsigned int i=0; i<ntoys; i++)
    {
      //S e t u p   p a r a m e t e r s   a n d   P D F
      //-----------------------------------------------
      //angle theta and the weight to account for the acceptance effect
      RooRealVar costheta("costheta","costheta", -1.0, 1.0);
      RooRealVar weight("weight","weight", 0.0, 1000.0);

      //initialise parameters to fit
      RooRealVar c0("c0","c0", c0gen, -1.0, 1.0);
      RooRealVar c1("c1","c1", c1gen, -1.0, 1.0);
      c0.setError(0.01);
      c1.setError(0.01);
      //create simple second order polynomial as probability density function
      RooPolynomial pol("pol", "pol", costheta, RooArgList(c0, c1), 1);

      //G e n e r a t e   d a t a   s e t   f o r   p s e u d o e x p e r i m e n t   i
      //-------------------------------------------------------------------------------
      RooDataSet data("data","data",RooArgSet(costheta, weight), WeightVar("weight"));
      //generate nstats events 
      for (unsigned int j=0; j<nstats; j++)
	{
	  bool finished = false;
	  //use simple accept/reject for generation
	  while (!finished)
	    {
	      costheta = 2.0*rnd->Rndm()-1.0;
	      //efficiency for the specific value of cos(theta)
	      double eff = 1.0;
	      if (acceptancemodel == 1)
		eff = 1.0 - 0.7 * costheta.getValV()*costheta.getValV();
	      else
		eff = 0.3 + 0.7 * costheta.getValV()*costheta.getValV();
	      //use 1/eff as weight to account for acceptance
	      weight = 1.0/eff;
	      //accept/reject
	      if (10.0*rnd->Rndm() < eff*pol.getValV())
		finished = true;
	    }
	  haccepted->Fill(costheta.getValV());
	  hweighted->Fill(costheta.getValV(), weight.getValV());
	  data.add(RooArgSet(costheta, weight), weight.getValV());
	}

      //F i t   t o y   u s i n g   t h e   t h r e e   d i f f e r e n t   a p p r o a c h e s   t o   u n c e r t a i n t y   d e t e r m i n a t i o n
      //-------------------------------------------------------------------------------------------------------------------------------------------------
      RooFitResult* result = pol.fitTo(data, Save(true), SumW2Error(false));//this uses the inverse weighted Hessian matrix
      hc0pull1->Fill((c0.getValV()-c0gen)/c0.getError());
      hc1pull1->Fill((c1.getValV()-c1gen)/c1.getError());
      
      result = pol.fitTo(data, Save(true), SumW2Error(true));//this uses the correction with the Hesse matrix with squared weights
      hc0pull2->Fill((c0.getValV()-c0gen)/c0.getError());
      hc1pull2->Fill((c1.getValV()-c1gen)/c1.getError());
      
      result = pol.fitTo(data, Save(true), AsymptoticError(true));//this uses the asymptotically correct approach
      hc0pull3->Fill((c0.getValV()-c0gen)/c0.getError());
      hc1pull3->Fill((c1.getValV()-c1gen)/c1.getError());      
    }
  
  // P l o t   o u t p u t   d i s t r i b u t i o n s
  //--------------------------------------------------
  
  //plot accepted (weighted) events
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);
  TCanvas* cevents = new TCanvas("cevents", "cevents", 800, 600);
  cevents->cd(1);
  hweighted->SetMinimum(0.0);
  hweighted->SetLineColor(2);
  hweighted->Draw("hist");
  haccepted->Draw("same hist");
  TLegend* leg = new TLegend(0.6, 0.8, 0.9, 0.9);
  leg->AddEntry(haccepted, "Accepted");
  leg->AddEntry(hweighted, "Weighted");
  leg->Draw();
  cevents->Update();
  
  //plot pull distributions
  TCanvas* cpull = new TCanvas("cpull", "cpull", 1200, 800);
  cpull->Divide(3,2);
  cpull->cd(1);
  gStyle->SetOptStat(1100);
  gStyle->SetOptFit(11);
  hc0pull1->Fit("gaus");
  hc0pull1->Draw("ep");
  cpull->cd(2);
  hc0pull2->Fit("gaus");
  hc0pull2->Draw("ep");
  cpull->cd(3);
  hc0pull3->Fit("gaus");
  hc0pull3->Draw("ep");
  cpull->cd(4);
  hc1pull1->Fit("gaus");
  hc1pull1->Draw("ep");
  cpull->cd(5);
  hc1pull2->Fit("gaus");
  hc1pull2->Draw("ep");
  cpull->cd(6);
  hc1pull3->Fit("gaus");
  hc1pull3->Draw("ep");
  cpull->Update();

  return 0;
}
