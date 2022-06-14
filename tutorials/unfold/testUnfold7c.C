/// \file
/// \ingroup tutorial_unfold
/// \notebook
/// Test program for the classes TUnfoldDensity and TUnfoldBinning.
///
/// A toy test of the TUnfold package
///
///
/// This example is documented in conference proceedings:
///
///   arXiv:1611.01927
///   12th Conference on Quark Confinement and the Hadron Spectrum (Confinement XII)
///
/// This is an example of unfolding a one-dimensional distribution. It compares
/// various unfolding methods:
///
///           matrix inversion, template fit, L-curve scan,
///           iterative unfolding, etc
///
/// Further details can  be found in talk by S.Schmitt at:
///
///   XII Quark Confinement and the Hadron Spectrum
///   29.8. - 3.9.2016	Thessaloniki, Greece
///   statictics session (+proceedings)
///
/// The example comprises several macros
///  - testUnfold7a.C   create root files with TTree objects for
///                     signal, background and data
///            - write files  testUnfold7_signal.root
///                           testUnfold7_background.root
///                           testUnfold7_data.root
///
///  - testUnfold7b.C   loop over trees and fill histograms based on the
///                     TUnfoldBinning objects
///            - read  testUnfold7binning.xml
///                    testUnfold7_signal.root
///                    testUnfold7_background.root
///                    testUnfold7_data.root
///
///            - write testUnfold7_histograms.root
///
///  - testUnfold7c.C   run the unfolding
///            - read  testUnfold7_histograms.root
///            - write testUnfold7_result.root
///            - write many histograms, to compare various unfolding methods
///
/// \macro_output
/// \macro_image
/// \macro_code
///
///  **Version 17.6, in parallel to changes in TUnfold**
///
///  This file is part of TUnfold.
///
///  TUnfold is free software: you can redistribute it and/or modify
///  it under the terms of the GNU General Public License as published by
///  the Free Software Foundation, either version 3 of the License, or
///  (at your option) any later version.
///
///  TUnfold is distributed in the hope that it will be useful,
///  but WITHOUT ANY WARRANTY; without even the implied warranty of
///  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///  GNU General Public License for more details.
///
///  You should have received a copy of the GNU General Public License
///  along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
///
/// \author Stefan Schmitt DESY, 14.10.2008

#include <iostream>
#include <cmath>
#include <map>
#include <TMath.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TFile.h>
#include <TROOT.h>
#include <TText.h>
#include <TLine.h>
#include <TLegend.h>
#include <TH1.h>
#include <TF1.h>
#include <TFitter.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TVectorD.h>
#include <TMatrixDSymEigen.h>
#include <TFitResult.h>
#include <TRandom3.h>
#include "TUnfoldDensity.h"

using namespace std;

// #define PRINT_MATRIX_L

#define TEST_INPUT_COVARIANCE

void CreateHistogramCopies(TH1 *h[3],TUnfoldBinning const *binning);
void CreateHistogramCopies(TH2 *h[3],TUnfoldBinning const *binningX);

TH2 *AddOverflowXY(TH2 *h,double widthX,double widthY);
TH1 *AddOverflowX(TH1 *h,double width);

void DrawOverflowX(TH1 *h,double posy);
void DrawOverflowY(TH1 *h,double posx);


double const kLegendFontSize=0.05;
int kNbinC=0;

void DrawPadProbability(TH2 *h);
void DrawPadEfficiency(TH1 *h);
void DrawPadReco(TH1 *histMcRec,TH1 *histMcbgrRec,TH1 *histDataRec,
                 TH1 *histDataUnfold,TH2 *histProbability,TH2 *histRhoij);
void DrawPadTruth(TH1 *histMcsigGen,TH1 *histDataGen,TH1 *histDataUnfold,
                  char const *text=0,double tau=0.0,vector<double> const *r=0,
                  TF1 *f=0);
void DrawPadCorrelations(TH2 *h,
                         vector<pair<TF1*,vector<double> > > const *table);

TFitResultPtr DoFit(TH1 *h,TH2 *rho,TH1 *truth,char const *text,
                    vector<pair<TF1*,vector<double> > > &table,int niter=0);

void GetNiterGraphs(int iFirst,int iLast,vector<pair<TF1*,
                    vector<double> > > const &table,int color,
                    TGraph *graph[4],int style);
void GetNiterHist(int ifit,vector<pair<TF1*,vector<double> > > const &table,
                  TH1 *hist[4],int color,int style,int fillStyle);

#ifdef WITH_IDS
void IDSfirst(TVectorD *data, TVectorD *dataErr, TMatrixD *A_, Double_t lambdaL_, TVectorD* &unfres1IDS_,TVectorD *&soustr);

void IDSiterate(TVectorD *data, TVectorD *dataErr, TMatrixD *A_,TMatrixD *Am_,
                Double_t lambdaU_, Double_t lambdaM_, Double_t lambdaS_,
                TVectorD* &unfres2IDS_ ,TVectorD *&soustr);
#endif

TRandom3 *g_rnd;

void testUnfold7c()
{
   gErrorIgnoreLevel=kInfo;
  // switch on histogram errors
  TH1::SetDefaultSumw2();

  gStyle->SetOptStat(0);

  g_rnd=new TRandom3(4711);

  //==============================================
  // step 1 : open output file
  TFile *outputFile=new TFile("testUnfold7_results.root","recreate");

  //==============================================
  // step 2 : read binning schemes and input histograms
  TFile *inputFile=new TFile("testUnfold7_histograms.root");

  outputFile->cd();

  TUnfoldBinning *fineBinning,*coarseBinning;

  inputFile->GetObject("fine",fineBinning);
  inputFile->GetObject("coarse",coarseBinning);

  if((!fineBinning)||(!coarseBinning)) {
     cout<<"problem to read binning schemes\n";
  }

  // save binning schemes to output file
  fineBinning->Write();
  coarseBinning->Write();

  // read histograms
#define READ(TYPE,binning,name)                       \
  TYPE *name[3]; inputFile->GetObject(#name,name[0]); \
  name[0]->Write();                                   \
  if(!name[0]) cout<<"Error reading " #name "\n";     \
  CreateHistogramCopies(name,binning);

  outputFile->cd();

  READ(TH1,fineBinning,histDataRecF);
  READ(TH1,coarseBinning,histDataRecC);
  READ(TH1,fineBinning,histDataBgrF);
  READ(TH1,coarseBinning,histDataBgrC);
  READ(TH1,coarseBinning,histDataGen);

  READ(TH2,fineBinning,histMcsigGenRecF);
  READ(TH2,coarseBinning,histMcsigGenRecC);
  READ(TH1,fineBinning,histMcsigRecF);
  READ(TH1,coarseBinning,histMcsigRecC);
  READ(TH1,coarseBinning,histMcsigGen);

  READ(TH1,fineBinning,histMcbgrRecF);
  READ(TH1,coarseBinning,histMcbgrRecC);

  TH1 *histOutputCtau0[3];
  TH2 *histRhoCtau0;
  TH1 *histOutputCLCurve[3];
  //TH2 *histRhoCLCurve;
  TH2 *histProbC;
  double tauMin=1.e-4;
  double tauMax=1.e-1;
  double fBgr=1.0; // 0.2/0.25;
  double biasScale=1.0;
  TUnfold::ERegMode mode=TUnfold::kRegModeSize; //Derivative;

  //double tauC;
  {
  TUnfoldDensity *tunfoldC=
     new TUnfoldDensity(histMcsigGenRecC[0],
                        TUnfold::kHistMapOutputHoriz,
                        mode,
                        TUnfold::kEConstraintNone,//Area,
                        TUnfoldDensity::kDensityModeNone,
                        coarseBinning,
                        coarseBinning);
  tunfoldC->SetInput(histDataRecC[0],biasScale);
  tunfoldC->SubtractBackground(histMcbgrRecC[0],"BGR",fBgr,0.0);
  tunfoldC->DoUnfold(0.);
  histOutputCtau0[0]=tunfoldC->GetOutput("histOutputCtau0");
  histRhoCtau0=tunfoldC->GetRhoIJtotal("histRhoCtau0");
  CreateHistogramCopies(histOutputCtau0,coarseBinning);
  tunfoldC->ScanLcurve(50,tauMin,tauMax,0);
  /* tauC= */tunfoldC->GetTau();
  //tunfoldC->ScanTau(50,1.E-7,1.E-1,0,TUnfoldDensity::kEScanTauRhoAvg);
  histOutputCLCurve[0]=tunfoldC->GetOutput("histOutputCLCurve");
  /* histRhoCLCurve= */tunfoldC->GetRhoIJtotal("histRhoCLCurve");
  CreateHistogramCopies(histOutputCLCurve,coarseBinning);
  histProbC=tunfoldC->GetProbabilityMatrix("histProbC",";P_T(gen);P_T(rec)");
  }
  TH1 *histOutputFtau0[3];
  TH2 *histRhoFtau0;
  TH1 *histOutputFLCurve[3];
  //TH2 *histRhoFLCurve;
  TH2 *histProbF;
  TGraph *lCurve;
  TSpline *logTauX,*logTauY;
  tauMin=3.E-4;
  tauMax=3.E-2;
  //double tauF;
  {
  TUnfoldDensity *tunfoldF=
     new TUnfoldDensity(histMcsigGenRecF[0],
                        TUnfold::kHistMapOutputHoriz,
                        mode,
                        TUnfold::kEConstraintNone,//Area,
                        TUnfoldDensity::kDensityModeNone,
                        coarseBinning,
                        fineBinning);
  tunfoldF->SetInput(histDataRecF[0],biasScale);
  tunfoldF->SubtractBackground(histMcbgrRecF[0],"BGR",fBgr,0.0);
  tunfoldF->DoUnfold(0.);
  histOutputFtau0[0]=tunfoldF->GetOutput("histOutputFtau0");
  histRhoFtau0=tunfoldF->GetRhoIJtotal("histRhoFtau0");
  CreateHistogramCopies(histOutputFtau0,coarseBinning);
  tunfoldF->ScanLcurve(50,tauMin,tauMax,0);
  //tunfoldF->DoUnfold(tauC);
  /* tauF= */tunfoldF->GetTau();
  //tunfoldF->ScanTau(50,1.E-7,1.E-1,0,TUnfoldDensity::kEScanTauRhoAvg);
  histOutputFLCurve[0]=tunfoldF->GetOutput("histOutputFLCurve");
  /* histRhoFLCurve= */tunfoldF->GetRhoIJtotal("histRhoFLCurve");
  CreateHistogramCopies(histOutputFLCurve,coarseBinning);
  histProbF=tunfoldF->GetProbabilityMatrix("histProbF",";P_T(gen);P_T(rec)");
  }
  TH1 *histOutputFAtau0[3];
  TH2 *histRhoFAtau0;
  TH1 *histOutputFALCurve[3];
  TH2 *histRhoFALCurve;
  TH1 *histOutputFArho[3];
  TH2 *histRhoFArho;
  TSpline *rhoScan=0;
  TSpline *logTauCurvature=0;

  double tauFA,tauFArho;
  {
  TUnfoldDensity *tunfoldFA=
     new TUnfoldDensity(histMcsigGenRecF[0],
                        TUnfold::kHistMapOutputHoriz,
                        mode,
                        TUnfold::kEConstraintArea,
                        TUnfoldDensity::kDensityModeNone,
                        coarseBinning,
                        fineBinning);
  tunfoldFA->SetInput(histDataRecF[0],biasScale);
  tunfoldFA->SubtractBackground(histMcbgrRecF[0],"BGR",fBgr,0.0);
  tunfoldFA->DoUnfold(0.);
  histOutputFAtau0[0]=tunfoldFA->GetOutput("histOutputFAtau0");
  histRhoFAtau0=tunfoldFA->GetRhoIJtotal("histRhoFAtau0");
  CreateHistogramCopies(histOutputFAtau0,coarseBinning);
  tunfoldFA->ScanTau(50,tauMin,tauMax,&rhoScan,TUnfoldDensity::kEScanTauRhoAvg);
  tauFArho=tunfoldFA->GetTau();
  histOutputFArho[0]=tunfoldFA->GetOutput("histOutputFArho");
  histRhoFArho=tunfoldFA->GetRhoIJtotal("histRhoFArho");
  CreateHistogramCopies(histOutputFArho,coarseBinning);

  tunfoldFA->ScanLcurve(50,tauMin,tauMax,&lCurve,&logTauX,&logTauY,&logTauCurvature);
  tauFA=tunfoldFA->GetTau();
  histOutputFALCurve[0]=tunfoldFA->GetOutput("histOutputFALCurve");
  histRhoFALCurve=tunfoldFA->GetRhoIJtotal("histRhoFALCurve");
  CreateHistogramCopies(histOutputFALCurve,coarseBinning);
  }
  lCurve->Write();
  logTauX->Write();
  logTauY->Write();


  double widthC=coarseBinning->GetBinSize(histProbC->GetNbinsY()+1);
  double widthF=fineBinning->GetBinSize(histProbF->GetNbinsY()+1);

  TH2 *histProbCO=AddOverflowXY(histProbC,widthC,widthC);
  TH2 *histProbFO=AddOverflowXY(histProbF,widthC,widthF);

  // efficiency
  TH1 *histEfficiencyC=histProbCO->ProjectionX("histEfficiencyC");
  kNbinC=histProbCO->GetNbinsX();

  // reconstructed quantities with overflow (coarse binning)
  // MC: add signal and bgr
  TH1 *histMcsigRecCO=AddOverflowX(histMcsigRecC[2],widthC);
  TH1 *histMcbgrRecCO=AddOverflowX(histMcbgrRecC[2],widthC);
  histMcbgrRecCO->Scale(fBgr);
  TH1 *histMcRecCO=(TH1 *)histMcsigRecCO->Clone("histMcRecC0");
  histMcRecCO->Add(histMcsigRecCO,histMcbgrRecCO);
  TH1 *histDataRecCO=AddOverflowX(histDataRecC[2],widthC);
  //TH1 *histDataRecCNO=AddOverflowX(histDataRecC[1],widthC);

  TH1 *histMcsigRecFO=AddOverflowX(histMcsigRecF[2],widthF);
  TH1 *histMcbgrRecFO=AddOverflowX(histMcbgrRecF[2],widthF);
  histMcbgrRecFO->Scale(fBgr);
  TH1 *histMcRecFO=(TH1 *)histMcsigRecFO->Clone("histMcRecF0");
  histMcRecFO->Add(histMcsigRecFO,histMcbgrRecFO);
  TH1 *histDataRecFO=AddOverflowX(histDataRecF[2],widthF);

  // truth level with overflow
  TH1 *histMcsigGenO=AddOverflowX(histMcsigGen[2],widthC);
  TH1 *histDataGenO=AddOverflowX(histDataGen[2],widthC);

  // unfolding result with overflow
  TH1 *histOutputCtau0O=AddOverflowX(histOutputCtau0[2],widthC);
  TH2 *histRhoCtau0O=AddOverflowXY(histRhoCtau0,widthC,widthC);
  //TH1 *histOutputCLCurveO=AddOverflowX(histOutputCLCurve[2],widthC);
  //TH2 *histRhoCLCurveO=AddOverflowXY(histRhoCLCurve,widthC,widthC);
  TH1 *histOutputFtau0O=AddOverflowX(histOutputFtau0[2],widthC);
  TH2 *histRhoFtau0O=AddOverflowXY(histRhoFtau0,widthC,widthC);
  TH1 *histOutputFAtau0O=AddOverflowX(histOutputFAtau0[2],widthC);
  TH2 *histRhoFAtau0O=AddOverflowXY(histRhoFAtau0,widthC,widthC);
  //TH1 *histOutputFLCurveO=AddOverflowX(histOutputFLCurve[2],widthC);
  //TH2 *histRhoFLCurveO=AddOverflowXY(histRhoFLCurve,widthC,widthC);
  TH1 *histOutputFALCurveO=AddOverflowX(histOutputFALCurve[2],widthC);
  TH2 *histRhoFALCurveO=AddOverflowXY(histRhoFALCurve,widthC,widthC);
  TH1 *histOutputFArhoO=AddOverflowX(histOutputFArho[2],widthC);
  TH2 *histRhoFArhoO=AddOverflowXY(histRhoFArho,widthC,widthC);

  // bin-by-bin
  TH2 *histRhoBBBO=(TH2 *)histRhoCtau0O->Clone("histRhoBBBO");
  for(int i=1;i<=histRhoBBBO->GetNbinsX();i++) {
     for(int j=1;j<=histRhoBBBO->GetNbinsX();j++) {
        histRhoBBBO->SetBinContent(i,j,(i==j)?1.:0.);
     }
  }
  TH1 *histDataBgrsub=(TH1 *)histDataRecCO->Clone("histDataBgrsub");
  histDataBgrsub->Add(histMcbgrRecCO,-fBgr);
  TH1 *histOutputBBBO=(TH1 *)histDataBgrsub->Clone("histOutputBBBO");
  histOutputBBBO->Divide(histMcsigRecCO);
  histOutputBBBO->Multiply(histMcsigGenO);

  // iterative
  int niter=1000;
  cout<<"maximum number of iterations: "<<niter<<"\n";

  vector <TH1 *>histOutputAgo,histOutputAgorep;
  vector <TH2 *>histRhoAgo,histRhoAgorep;
  vector<int> nIter;
  histOutputAgo.push_back((TH1*)histMcsigGenO->Clone("histOutputAgo-1"));
  histOutputAgorep.push_back((TH1*)histMcsigGenO->Clone("histOutputAgorep-1"));
  histRhoAgo.push_back((TH2*)histRhoBBBO->Clone("histRhoAgo-1"));
  histRhoAgorep.push_back((TH2*)histRhoBBBO->Clone("histRhoAgorep-1"));
  nIter.push_back(-1);

  int nx=histProbCO->GetNbinsX();
  int ny=histProbCO->GetNbinsY();
  TMatrixD covAgo(nx+ny,nx+ny);
  TMatrixD A(ny,nx);
  TMatrixD AToverEps(nx,ny);
  for(int i=0;i<nx;i++) {
     double epsilonI=0.;
     for(int j=0;j<ny;j++) {
        epsilonI+= histProbCO->GetBinContent(i+1,j+1);
     }
     for(int j=0;j<ny;j++) {
        double aji=histProbCO->GetBinContent(i+1,j+1);
        A(j,i)=aji;
        AToverEps(i,j)=aji/epsilonI;
     }
  }
  for(int i=0;i<nx;i++) {
     covAgo(i,i)=TMath::Power
        (histOutputAgo[0]->GetBinError(i+1)
         *histOutputAgo[0]->GetXaxis()->GetBinWidth(i+1),2.);
  }
  for(int i=0;i<ny;i++) {
     covAgo(i+nx,i+nx)=TMath::Power
        (histDataRecCO->GetBinError(i+1)
         *histDataRecCO->GetXaxis()->GetBinWidth(i+1),2.);
  }
#define NREPLICA 300
  vector<TVectorD *> y(NREPLICA);
  vector<TVectorD *> yMb(NREPLICA);
  vector<TVectorD *> yErr(NREPLICA);
  vector<TVectorD *> x(NREPLICA);
  TVectorD b(nx);
  for(int nr=0;nr<NREPLICA;nr++) {
     x[nr]=new TVectorD(nx);
     y[nr]=new TVectorD(ny);
     yMb[nr]=new TVectorD(ny);
     yErr[nr]=new TVectorD(ny);
  }
  for(int i=0;i<nx;i++) {
     (*x[0])(i)=histOutputAgo[0]->GetBinContent(i+1)
        *histOutputAgo[0]->GetXaxis()->GetBinWidth(i+1);
     for(int nr=1;nr<NREPLICA;nr++) {
        (*x[nr])(i)=(*x[0])(i);
     }
  }
  for(int i=0;i<ny;i++) {
     (*y[0])(i)=histDataRecCO->GetBinContent(i+1)
        *histDataRecCO->GetXaxis()->GetBinWidth(i+1);
     for(int nr=1;nr<NREPLICA;nr++) {
        (*y[nr])(i)=g_rnd->Poisson((*y[0])(i));
        (*yErr[nr])(i)=TMath::Sqrt((*y[nr])(i));
     }
     b(i)=histMcbgrRecCO->GetBinContent(i+1)*
        histMcbgrRecCO->GetXaxis()->GetBinWidth(i+1);
     for(int nr=0;nr<NREPLICA;nr++) {
        (*yMb[nr])(i)=(*y[nr])(i)-b(i);
     }
  }
  for(int iter=0;iter<=niter;iter++) {
     if(!(iter %100)) cout<<iter<<"\n";
     for(int nr=0;nr<NREPLICA;nr++) {
        TVectorD yrec=A*(*x[nr])+b;
        TVectorD yOverYrec(ny);
        for(int j=0;j<ny;j++) {
           yOverYrec(j)=(*y[nr])(j)/yrec(j);
        }
        TVectorD f=AToverEps * yOverYrec;
        TVectorD xx(nx);
        for(int i=0;i<nx;i++) {
           xx(i) = (*x[nr])(i) * f(i);
        }
        if(nr==0) {
           TMatrixD xdf_dr=AToverEps;
           for(int i=0;i<nx;i++) {
              for(int j=0;j<ny;j++) {
                 xdf_dr(i,j) *= (*x[nr])(i);
              }
           }
           TMatrixD dr_dxdy(ny,nx+ny);
           for(int j=0;j<ny;j++) {
              dr_dxdy(j,nx+j)=1.0/yrec(j);
              for(int i=0;i<nx;i++) {
                 dr_dxdy(j,i)= -yOverYrec(j)/yrec(j)*A(j,i);
              }
           }
           TMatrixD dxy_dxy(nx+ny,nx+ny);
           dxy_dxy.SetSub(0,0,xdf_dr*dr_dxdy);
           for(int i=0;i<nx;i++) {
              dxy_dxy(i,i) +=f(i);
           }
           for(int i=0;i<ny;i++) {
              dxy_dxy(nx+i,nx+i) +=1.0;
           }
           TMatrixD VDT(covAgo,TMatrixD::kMultTranspose,dxy_dxy);
           covAgo= dxy_dxy*VDT;
        }
        (*x[nr])=xx;
     }
     if((iter<=25)||
        ((iter<=100)&&(iter %5==0))||
        ((iter<=1000)&&(iter %50==0))||
        (iter %1000==0)) {
        nIter.push_back(iter);
        TH1 * h=(TH1*)histOutputAgo[0]->Clone
           (TString::Format("histOutputAgo%d",iter));
        histOutputAgo.push_back(h);
        for(int i=0;i<nx;i++) {
           double bw=h->GetXaxis()->GetBinWidth(i+1);
           h->SetBinContent(i+1,(*x[0])(i)/bw);
           h->SetBinError(i+1,TMath::Sqrt(covAgo(i,i))/bw);
        }
        TH2 *h2=(TH2*)histRhoAgo[0]->Clone
           (TString::Format("histRhoAgo%d",iter));
        histRhoAgo.push_back(h2);
        for(int i=0;i<nx;i++) {
           for(int j=0;j<nx;j++) {
              double rho= covAgo(i,j)/TMath::Sqrt(covAgo(i,i)*covAgo(j,j));
              if((i!=j)&&(TMath::Abs(rho)>=1.0)) {
                 cout<<"bad error matrix: iter="<<iter<<"\n";
                 exit(0);
              }
              h2->SetBinContent(i+1,j+1,rho);
           }
        }
        // error and correlations from replica analysis
        h=(TH1*)histOutputAgo[0]->Clone
           (TString::Format("histOutputAgorep%d",iter));
        h2=(TH2*)histRhoAgo[0]->Clone
           (TString::Format("histRhoAgorep%d",iter));
        histOutputAgorep.push_back(h);
        histRhoAgorep.push_back(h2);

        TVectorD mean(nx);
        double w=1./(NREPLICA-1.);
        for(int nr=1;nr<NREPLICA;nr++) {
           mean += w* *x[nr];
        }
        TMatrixD covAgorep(nx,nx);
        for(int nr=1;nr<NREPLICA;nr++) {
           //TMatrixD dx= (*x)-mean;
           TMatrixD dx(nx,1);
           for(int i=0;i<nx;i++) {
              dx(i,0)= (*x[nr])(i)-(*x[0])(i);
           }
           covAgorep += w*TMatrixD(dx,TMatrixD::kMultTranspose,dx);
        }

        for(int i=0;i<nx;i++) {
           double bw=h->GetXaxis()->GetBinWidth(i+1);
           h->SetBinContent(i+1,(*x[0])(i)/bw);
           h->SetBinError(i+1,TMath::Sqrt(covAgorep(i,i))/bw);
           // cout<<i<<" "<<(*x[0])(i)/bw<<" +/-"<<TMath::Sqrt(covAgorep(i,i))/bw<<" "<<TMath::Sqrt(covAgo(i,i))/bw<<"\n";
        }
        for(int i=0;i<nx;i++) {
           for(int j=0;j<nx;j++) {
              double rho= covAgorep(i,j)/
                 TMath::Sqrt(covAgorep(i,i)*covAgorep(j,j));
              if((i!=j)&&(TMath::Abs(rho)>=1.0)) {
                 cout<<"bad error matrix: iter="<<iter<<"\n";
                 exit(0);
              }
              h2->SetBinContent(i+1,j+1,rho);
           }
        }
     }
  }

#ifdef WITH_IDS
  // IDS Malaescu
  int niterIDS=100;
  vector<TVectorD*> unfresIDS(NREPLICA),soustr(NREPLICA);
  cout<<"IDS number of iterations: "<<niterIDS<<"\n";
  TMatrixD *Am_IDS[NREPLICA];
  TMatrixD A_IDS(ny,nx);
  for(int nr=0;nr<NREPLICA;nr++) {
     Am_IDS[nr]=new TMatrixD(ny,nx);
  }
  for(int iy=0;iy<ny;iy++) {
     for(int ix=0;ix<nx;ix++) {
        A_IDS(iy,ix)=histMcsigGenRecC[0]->GetBinContent(ix+1,iy+1);
     }
  }
  double lambdaL=0.;
  Double_t lambdaUmin = 1.0000002;
  Double_t lambdaMmin = 0.0000001;
  Double_t lambdaS = 0.000001;
  double lambdaU=lambdaUmin;
  double lambdaM=lambdaMmin;
  vector<TH1 *> histOutputIDS;
  vector<TH2 *> histRhoIDS;
  histOutputIDS.push_back((TH1*)histOutputAgo[0]->Clone("histOutputIDS-1"));
  histRhoIDS.push_back((TH2*)histRhoAgo[0]->Clone("histRhoIDS-1"));
  histOutputIDS.push_back((TH1*)histOutputAgo[0]->Clone("histOutputIDS0"));
  histRhoIDS.push_back((TH2*)histRhoAgo[0]->Clone("histRhoIDS0"));
  for(int iter=1;iter<=niterIDS;iter++) {
     if(!(iter %10)) cout<<iter<<"\n";

     for(int nr=0;nr<NREPLICA;nr++) {
        if(iter==1) {
           IDSfirst(yMb[nr],yErr[nr],&A_IDS,lambdaL,unfresIDS[nr],soustr[nr]);
        } else {
           IDSiterate(yMb[nr],yErr[nr],&A_IDS,Am_IDS[nr],
                      lambdaU,lambdaM,lambdaS,
                      unfresIDS[nr],soustr[nr]);
        }
     }
     unsigned ix;
     for(ix=0;ix<nIter.size();ix++) {
        if(nIter[ix]==iter) break;
     }
     if(ix<nIter.size()) {
        TH1 * h=(TH1*)histOutputIDS[0]->Clone
           (TString::Format("histOutputIDS%d",iter));
        TH2 *h2=(TH2*)histRhoIDS[0]->Clone
           (TString::Format("histRhoIDS%d",iter));
        histOutputIDS.push_back(h);
        histRhoIDS.push_back(h2);
        TVectorD mean(nx);
        double w=1./(NREPLICA-1.);
        for(int nr=1;nr<NREPLICA;nr++) {
           mean += w* (*unfresIDS[nr]);
        }
        TMatrixD covIDSrep(nx,nx);
        for(int nr=1;nr<NREPLICA;nr++) {
           //TMatrixD dx= (*x)-mean;
           TMatrixD dx(nx,1);
           for(int i=0;i<nx;i++) {
              dx(i,0)= (*unfresIDS[nr])(i)-(*unfresIDS[0])(i);
           }
           covIDSrep += w*TMatrixD(dx,TMatrixD::kMultTranspose,dx);
        }
        for(int i=0;i<nx;i++) {
           double bw=h->GetXaxis()->GetBinWidth(i+1);
           h->SetBinContent(i+1,(*unfresIDS[0])(i)/bw/
                            histEfficiencyC->GetBinContent(i+1));
           h->SetBinError(i+1,TMath::Sqrt(covIDSrep(i,i))/bw/
                          histEfficiencyC->GetBinContent(i+1));
           // cout<<i<<" "<<(*x[0])(i)/bw<<" +/-"<<TMath::Sqrt(covAgorep(i,i))/bw<<" "<<TMath::Sqrt(covAgo(i,i))/bw<<"\n";
        }
        for(int i=0;i<nx;i++) {
           for(int j=0;j<nx;j++) {
              double rho= covIDSrep(i,j)/
                 TMath::Sqrt(covIDSrep(i,i)*covIDSrep(j,j));
              if((i!=j)&&(TMath::Abs(rho)>=1.0)) {
                 cout<<"bad error matrix: iter="<<iter<<"\n";
                 exit(0);
              }
              h2->SetBinContent(i+1,j+1,rho);
           }
        }
     }
  }
#endif

  //double NEdSmc=histDataBgrsub->GetSumOfWeights();

  vector<pair<TF1 *,vector<double> > > table;

  TCanvas *c1=new TCanvas("c1","",600,600);
  TCanvas *c2sq=new TCanvas("c2sq","",600,600);
  c2sq->Divide(1,2);
  TCanvas *c2w=new TCanvas("c2w","",600,300);
  c2w->Divide(2,1);
  TCanvas *c4=new TCanvas("c4","",600,600);
  c4->Divide(2,2);
  //TCanvas *c3n=new TCanvas("c3n","",600,600);
  TPad *subn[3];
  //gROOT->SetStyle("xTimes2");
  subn[0]= new TPad("subn0","",0.,0.5,1.,1.);
  //gROOT->SetStyle("square");
  subn[1]= new TPad("subn1","",0.,0.,0.5,0.5);
  subn[2]= new TPad("subn2","",0.5,0.0,1.,0.5);
  for(int i=0;i<3;i++) {
     subn[i]->SetFillStyle(0);
     subn[i]->Draw();
  }
  TCanvas *c3c=new TCanvas("c3c","",600,600);
  TPad *subc[3];
  //gROOT->SetStyle("xTimes2");
  subc[0]= new TPad("sub0","",0.,0.5,1.,1.);
  //gROOT->SetStyle("squareCOLZ");
  subc[1]= new TPad("sub1","",0.,0.,0.5,0.5);
  //gROOT->SetStyle("square");
  subc[2]= new TPad("sub2","",0.5,0.0,1.,0.5);
  for(int i=0;i<3;i++) {
     subc[i]->SetFillStyle(0);
     subc[i]->Draw();
  }

  //=========================== example ==================================

  c2w->cd(1);
  DrawPadTruth(histMcsigGenO,histDataGenO,0);
  c2w->cd(2);
  DrawPadReco(histMcRecCO,histMcbgrRecCO,histDataRecCO,0,0,0);
  c2w->SaveAs("exampleTR.eps");

  //=========================== example ==================================

  c2w->cd(1);
  DrawPadProbability(histProbCO);
  c2w->cd(2);
  DrawPadEfficiency(histEfficiencyC);
  c2w->SaveAs("exampleAE.eps");

  int iFitInversion=table.size();
  DoFit(histOutputCtau0O,histRhoCtau0O,histDataGenO,"inversion",table);
  //=========================== inversion ==================================

  subc[0]->cd();
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputCtau0O,"inversion",0.,
               &table[table.size()-1].second);
  subc[1]->cd();
  DrawPadCorrelations(histRhoCtau0O,&table);
  subc[2]->cd();
  DrawPadReco(histMcRecCO,histMcbgrRecCO,histDataRecCO,
              histOutputCtau0O,histProbCO,histRhoCtau0O);
  c3c->SaveAs("inversion.eps");


  DoFit(histOutputFtau0O,histRhoFtau0O,histDataGenO,"template",table);
  //=========================== template ==================================

  subc[0]->cd();
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputFtau0O,"fit",0.,
               &table[table.size()-1].second);
  subc[1]->cd();
  DrawPadCorrelations(histRhoFtau0O,&table);
  subc[2]->cd();
  DrawPadReco(histMcRecFO,histMcbgrRecFO,histDataRecFO,
              histOutputFtau0O,histProbFO,histRhoFtau0O);
  c3c->SaveAs("template.eps");

  DoFit(histOutputFAtau0O,histRhoFAtau0O,histDataGenO,"template+area",table);
  //=========================== template+area ==================================

  subc[0]->cd();
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputFAtau0O,"fit",0.,
               &table[table.size()-1].second);
  subc[1]->cd();
  DrawPadCorrelations(histRhoFAtau0O,&table);
  subc[2]->cd();
  DrawPadReco(histMcRecFO,histMcbgrRecFO,histDataRecFO,
              histOutputFAtau0O,histProbFO,histRhoFAtau0O);
  c3c->SaveAs("templateA.eps");

  int iFitFALCurve=table.size();
  DoFit(histOutputFALCurveO,histRhoFALCurveO,histDataGenO,"Tikhonov+area",table);
  //=========================== template+area+tikhonov =====================
  subc[0]->cd();
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputFALCurveO,"Tikhonov",tauFA,
                &table[table.size()-1].second);
  subc[1]->cd();
  DrawPadCorrelations(histRhoFALCurveO,&table);
  subc[2]->cd();
  DrawPadReco(histMcRecFO,histMcbgrRecFO,histDataRecFO,
              histOutputFALCurveO,histProbFO,histRhoFALCurveO);
  c3c->SaveAs("lcurveFA.eps");

  int iFitFArho=table.size();
  DoFit(histOutputFArhoO,histRhoFArhoO,histDataGenO,"min(rhomax)",table);
  //=========================== template+area+tikhonov =====================
  subc[0]->cd();
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputFArhoO,"Tikhonov",tauFArho,
                &table[table.size()-1].second);
  subc[1]->cd();
  DrawPadCorrelations(histRhoFArho,&table);
  subc[2]->cd();
  DrawPadReco(histMcRecFO,histMcbgrRecFO,histDataRecFO,
              histOutputFArhoO,histProbFO,histRhoFArhoO);
  c3c->SaveAs("rhoscanFA.eps");

  int iFitBinByBin=table.size();
  DoFit(histOutputBBBO,histRhoBBBO,histDataGenO,"bin-by-bin",table);
  //=========================== bin-by-bin =================================
  //c->cd(1);
  //DrawPadProbability(histProbCO);
  //c->cd(2);
  //DrawPadCorrelations(histRhoBBBO,&table);
  c2sq->cd(1);
  DrawPadReco(histMcRecCO,histMcbgrRecCO,histDataRecCO,
              histOutputBBBO,histProbCO,histRhoBBBO);
  c2sq->cd(2);
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputBBBO,"bin-by-bin",0.,
               &table[table.size()-1].second);
  c2sq->SaveAs("binbybin.eps");


  //=========================== iterative ===================================
  int iAgoFirstFit=table.size();
  for(size_t i=1;i<histRhoAgorep.size();i++) {
     int n=nIter[i];
     bool isFitted=false;
     DoFit(histOutputAgorep[i],histRhoAgorep[i],histDataGenO,
           TString::Format("iterative, N=%d",n),table,n);
     isFitted=true;
     subc[0]->cd();
     DrawPadTruth(histMcsigGenO,histDataGenO,histOutputAgorep[i],
                  TString::Format("iterative N=%d",nIter[i]),0.,
                  isFitted ? &table[table.size()-1].second : 0);
     subc[1]->cd();
     DrawPadCorrelations(histRhoAgorep[i],&table);
     subc[2]->cd();
     DrawPadReco(histMcRecCO,histMcbgrRecCO,histDataRecCO,
                 histOutputAgorep[i],histProbCO,histRhoAgorep[i]);
     c3c->SaveAs(TString::Format("iterative%d.eps",nIter[i]));
  }
  int iAgoLastFit=table.size();

#ifdef WITH_IDS
  int iIDSFirstFit=table.size();

  //=========================== IDS ===================================

  for(size_t i=2;i<histRhoIDS.size();i++) {
     int n=nIter[i];
     bool isFitted=false;
     DoFit(histOutputIDS[i],histRhoIDS[i],histDataGenO,
           TString::Format("IDS, N=%d",n),table,n);
     isFitted=true;
     subc[0]->cd();
     DrawPadTruth(histMcsigGenO,histDataGenO,histOutputIDS[i],
                  TString::Format("IDS N=%d",nIter[i]),0.,
                  isFitted ? &table[table.size()-1].second : 0);
     subc[1]->cd();
     DrawPadCorrelations(histRhoIDS[i],&table);
     subc[2]->cd();
     DrawPadReco(histMcRecCO,histMcbgrRecCO,histDataRecCO,
                 histOutputIDS[i],histProbCO,histRhoIDS[i]);
     c3c->SaveAs(TString::Format("ids%d.eps",nIter[i]));
  }
  int iIDSLastFit=table.size();
#endif

  int nfit=table.size();
  TH1D *fitChindf=new TH1D("fitChindf",";algorithm;#chi^{2}/NDF",nfit,0,nfit);
  TH1D *fitNorm=new TH1D("fitNorm",";algorithm;Landau amplitude [1/GeV]",nfit,0,nfit);
  TH1D *fitMu=new TH1D("fitMu",";algorithm;Landau #mu [GeV]",nfit,0,nfit);
  TH1D *fitSigma=new TH1D("fitSigma",";algorithm;Landau #sigma [GeV]",nfit,0,nfit);
  for(int fit=0;fit<nfit;fit++) {
     TF1 *f=table[fit].first;
     vector<double> const &r=table[fit].second;
     fitChindf->GetXaxis()->SetBinLabel(fit+1,f->GetName());
     fitNorm->GetXaxis()->SetBinLabel(fit+1,f->GetName());
     fitMu->GetXaxis()->SetBinLabel(fit+1,f->GetName());
     fitSigma->GetXaxis()->SetBinLabel(fit+1,f->GetName());
     double chi2=r[0];
     double ndf=r[1];
     fitChindf->SetBinContent(fit+1,chi2/ndf);
     fitChindf->SetBinError(fit+1,TMath::Sqrt(2./ndf));
     fitNorm->SetBinContent(fit+1,f->GetParameter(0));
     fitNorm->SetBinError(fit+1,f->GetParError(0));
     fitMu->SetBinContent(fit+1,f->GetParameter(1));
     fitMu->SetBinError(fit+1,f->GetParError(1));
     fitSigma->SetBinContent(fit+1,f->GetParameter(2));
     fitSigma->SetBinError(fit+1,f->GetParError(2));
     cout<<"\""<<f->GetName()<<"\","<<r[2]/r[3]<<","<<r[3]
         <<","<<TMath::Prob(r[2],r[3])
         <<","<<chi2/ndf
         <<","<<ndf
         <<","<<TMath::Prob(r[0],r[1])
         <<","<<r[5];
     for(int i=1;i<3;i++) {
        cout<<","<<f->GetParameter(i)<<",\"\302\261\","<<f->GetParError(i);
     }
     cout<<"\n";
  }

  //=========================== L-curve ==========================
  c4->cd(1);
  lCurve->SetTitle("L curve;log_{10} L_{x};log_{10} L_{y}");
  lCurve->SetLineColor(kRed);
  lCurve->Draw("AL");
  c4->cd(2);
  gPad->Clear();
  c4->cd(3);
  logTauX->SetTitle(";log_{10} #tau;log_{10} L_{x}");
  logTauX->SetLineColor(kBlue);
  logTauX->Draw();
  c4->cd(4);
  logTauY->SetTitle(";log_{10} #tau;log_{10} L_{y}");
  logTauY->SetLineColor(kBlue);
  logTauY->Draw();
  c4->SaveAs("lcurveL.eps");

  //========================= rho and L-curve scan ===============
  c4->cd(1);
  logTauCurvature->SetTitle(";log_{10}(#tau);L curve curvature");
  logTauCurvature->SetLineColor(kRed);
  logTauCurvature->Draw();
  c4->cd(2);
  rhoScan->SetTitle(";log_{10}(#tau);average(#rho_{i})");
  rhoScan->SetLineColor(kRed);
  rhoScan->Draw();
  c4->cd(3);
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputFALCurveO,"Tikhonov",tauFA,
                &table[iFitFALCurve].second);
  c4->cd(4);
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputFArhoO,"Tikhonov",tauFArho,
                &table[iFitFArho].second);

  c4->SaveAs("scanTau.eps");


  TGraph *graphNiterAgoBay[4];
  GetNiterGraphs(iAgoFirstFit,iAgoFirstFit+1,table,kRed-2,graphNiterAgoBay,20);
  TGraph *graphNiterAgo[4];
  GetNiterGraphs(iAgoFirstFit,iAgoLastFit,table,kRed,graphNiterAgo,24);
#ifdef WITH_IDS
  TGraph *graphNiterIDS[4];
  GetNiterGraphs(iIDSFirstFit,iIDSLastFit,table,kMagenta,graphNiterIDS,21);
#endif
  TH1 *histNiterInversion[4];
  GetNiterHist(iFitInversion,table,histNiterInversion,kCyan,1,1001);
  TH1 *histNiterFALCurve[4];
  GetNiterHist(iFitFALCurve,table,histNiterFALCurve,kBlue,2,3353);
  TH1 *histNiterFArho[4];
  GetNiterHist(iFitFArho,table,histNiterFArho,kAzure-4,3,3353);
  TH1 *histNiterBinByBin[4];
  GetNiterHist(iFitBinByBin,table,histNiterBinByBin,kOrange+1,3,3335);

  histNiterInversion[0]->GetYaxis()->SetRangeUser(0.3,500.);
  histNiterInversion[1]->GetYaxis()->SetRangeUser(-0.1,0.9);
  histNiterInversion[2]->GetYaxis()->SetRangeUser(5.6,6.3);
  histNiterInversion[3]->GetYaxis()->SetRangeUser(1.6,2.4);

  TLine *line=0;
  c1->cd();
  for(int i=0;i<2;i++) {
     gPad->Clear();
     gPad->SetLogx();
     gPad->SetLogy((i<1));
     if(! histNiterInversion[i]) continue;
     histNiterInversion[i]->Draw("][");
     histNiterFALCurve[i]->Draw("SAME ][");
     histNiterFArho[i]->Draw("SAME ][");
     histNiterBinByBin[i]->Draw("SAME ][");
     graphNiterAgo[i]->Draw("LP");
     graphNiterAgoBay[i]->SetMarkerStyle(20);
     graphNiterAgoBay[i]->Draw("P");
#ifdef WITH_IDS
     graphNiterIDS[i]->Draw("LP");
#endif
     TLegend *legend;
     if(i==1) {
        legend=new TLegend(0.48,0.28,0.87,0.63);
     } else {
        legend=new TLegend(0.45,0.5,0.88,0.88);
     }
     legend->SetBorderSize(0);
     legend->SetFillStyle(1001);
     legend->SetFillColor(kWhite);
     legend->SetTextSize(kLegendFontSize*0.7);
     legend->AddEntry( histNiterInversion[0],"inversion","l");
     legend->AddEntry( histNiterFALCurve[0],"Tikhonov L-curve","l");
     legend->AddEntry( histNiterFArho[0],"Tikhonov global cor.","l");
     legend->AddEntry( histNiterBinByBin[0],"bin-by-bin","l");
     legend->AddEntry( graphNiterAgoBay[0],"\"Bayesian\"","p");
     legend->AddEntry( graphNiterAgo[0],"iterative","p");
#ifdef WITH_IDS
     legend->AddEntry( graphNiterIDS[0],"IDS","p");
#endif
     legend->Draw();

     c1->SaveAs(TString::Format("niter%d.eps",i));
  }

  //c4->cd(1);
  //DrawPadCorrelations(histRhoFALCurveO,&table);
  c2sq->cd(1);
  DrawPadTruth(histMcsigGenO,histDataGenO,histOutputFALCurveO,"Tikhonov",tauFA,
               &table[iFitFALCurve].second,table[iFitFALCurve].first);

  c2sq->cd(2);
  gPad->SetLogx();
  gPad->SetLogy(false);
  histNiterInversion[3]->DrawClone("E2");
  histNiterInversion[3]->SetFillStyle(0);
  histNiterInversion[3]->Draw("SAME HIST ][");
  histNiterFALCurve[3]->DrawClone("SAME E2");
  histNiterFALCurve[3]->SetFillStyle(0);
  histNiterFALCurve[3]->Draw("SAME HIST ][");
  histNiterFArho[3]->DrawClone("SAME E2");
  histNiterFArho[3]->SetFillStyle(0);
  histNiterFArho[3]->Draw("SAME HIST ][");
  histNiterBinByBin[3]->DrawClone("SAME E2");
  histNiterBinByBin[3]->SetFillStyle(0);
  histNiterBinByBin[3]->Draw("SAME HIST ][");
  double yTrue=1.8;
  line=new TLine(histNiterInversion[3]->GetXaxis()->GetXmin(),
                 yTrue,
                 histNiterInversion[3]->GetXaxis()->GetXmax(),
                 yTrue);
  line->SetLineWidth(3);
  line->Draw();

  graphNiterAgo[3]->Draw("LP");
  graphNiterAgoBay[3]->SetMarkerStyle(20);
  graphNiterAgoBay[3]->Draw("P");
#ifdef WITH_IDS
  graphNiterIDS[3]->Draw("LP");
#endif

  TLegend *legend;
  legend=new TLegend(0.55,0.53,0.95,0.97);
  legend->SetBorderSize(0);
  legend->SetFillStyle(1001);
  legend->SetFillColor(kWhite);
  legend->SetTextSize(kLegendFontSize);
  legend->AddEntry( line,"truth","l");
  legend->AddEntry( histNiterInversion[3],"inversion","l");
  legend->AddEntry( histNiterFALCurve[3],"Tikhonov L-curve","l");
  legend->AddEntry( histNiterFArho[3],"Tikhonov global cor.","l");
  legend->AddEntry( histNiterBinByBin[3],"bin-by-bin","l");
  legend->AddEntry( graphNiterAgoBay[3],"\"Bayesian\"","p");
  legend->AddEntry( graphNiterAgo[3],"iterative","p");
#ifdef WITH_IDS
  legend->AddEntry( graphNiterIDS[3],"IDS","p");
#endif
  legend->Draw();

  c2sq->SaveAs("fitSigma.eps");

  outputFile->Write();

  delete outputFile;

}

void GetNiterGraphs(int iFirst,int iLast,vector<pair<TF1*,
                    vector<double> > > const &table,int color,
                    TGraph *graph[4],int style) {
   TVectorD niter(iLast-iFirst);
   TVectorD eniter(iLast-iFirst);
   TVectorD chi2(iLast-iFirst);
   TVectorD gcor(iLast-iFirst);
   TVectorD mean(iLast-iFirst);
   TVectorD emean(iLast-iFirst);
   TVectorD sigma(iLast-iFirst);
   TVectorD esigma(iLast-iFirst);
   for(int ifit=iFirst;ifit<iLast;ifit++) {
      vector<double> const &r=table[ifit].second;
      niter(ifit-iFirst)=r[4];
      chi2(ifit-iFirst)=r[2]/r[3];
      gcor(ifit-iFirst)=r[5];
      TF1 const *f=table[ifit].first;
      mean(ifit-iFirst)=f->GetParameter(1);
      emean(ifit-iFirst)=f->GetParError(1);
      sigma(ifit-iFirst)=f->GetParameter(2);
      esigma(ifit-iFirst)=f->GetParError(2);
   }
   graph[0]=new TGraph(niter,chi2);
   graph[1]=new TGraph(niter,gcor);
   graph[2]=new TGraphErrors(niter,mean,eniter,emean);
   graph[3]=new TGraphErrors(niter,sigma,eniter,esigma);
   for(int g=0;g<4;g++) {
      if(graph[g]) {
         graph[g]->SetLineColor(color);
         graph[g]->SetMarkerColor(color);
         graph[g]->SetMarkerStyle(style);
      }
   }
}

void GetNiterHist(int ifit,vector<pair<TF1*,vector<double> > > const &table,
                  TH1 *hist[4],int color,int style,int fillStyle) {
   vector<double> const &r=table[ifit].second;
   TF1 const *f=table[ifit].first;
   hist[0]=new TH1D(table[ifit].first->GetName()+TString("_chi2"),
                    ";iteration;unfold-truth #chi^{2}/N_{D.F.}",1,0.2,1500.);
   hist[0]->SetBinContent(1,r[2]/r[3]);

   hist[1]=new TH1D(table[ifit].first->GetName()+TString("_gcor"),
                    ";iteration;avg(#rho_{i})",1,0.2,1500.);
   hist[1]->SetBinContent(1,r[5]);
   hist[2]=new TH1D(table[ifit].first->GetName()+TString("_mu"),
                     ";iteration;parameter #mu",1,0.2,1500.);
   hist[2]->SetBinContent(1,f->GetParameter(1));
   hist[2]->SetBinError(1,f->GetParError(1));
   hist[3]=new TH1D(table[ifit].first->GetName()+TString("_sigma"),
                    ";iteration;parameter #sigma",1,0.2,1500.);
   hist[3]->SetBinContent(1,f->GetParameter(2));
   hist[3]->SetBinError(1,f->GetParError(2));
   for(int h=0;h<4;h++) {
      if(hist[h]) {
         hist[h]->SetLineColor(color);
         hist[h]->SetLineStyle(style);
         if( hist[h]->GetBinError(1)>0.0) {
            hist[h]->SetFillColor(color-10);
            hist[h]->SetFillStyle(fillStyle);
         }
         hist[h]->SetMarkerStyle(0);
      }
   }
}

void CreateHistogramCopies(TH1 *h[3],TUnfoldBinning const *binning) {
   TString baseName(h[0]->GetName());
   Int_t *binMap;
   h[1]=binning->CreateHistogram(baseName+"_axis",kTRUE,&binMap);
   h[2]=(TH1 *)h[1]->Clone(baseName+"_binw");
   Int_t nMax=binning->GetEndBin()+1;
   for(Int_t iSrc=0;iSrc<nMax;iSrc++) {
      Int_t iDest=binMap[iSrc];
      double c=h[0]->GetBinContent(iSrc)+h[1]->GetBinContent(iDest);
      double e=TMath::Hypot(h[0]->GetBinError(iSrc),h[1]->GetBinError(iDest));
      h[1]->SetBinContent(iDest,c);
      h[1]->SetBinError(iDest,e);
      h[2]->SetBinContent(iDest,c);
      h[2]->SetBinError(iDest,e);
   }
   for(int iDest=0;iDest<=h[2]->GetNbinsX()+1;iDest++) {
      double c=h[2]->GetBinContent(iDest);
      double e=h[2]->GetBinError(iDest);
      double bw=binning->GetBinSize(iDest);
      /* if(bw!=h[2]->GetBinWidth(iDest)) {
         cout<<"bin "<<iDest<<" width="<<bw<<" "<<h[2]->GetBinWidth(iDest)
             <<"\n";
             } */
      if(bw>0.0) {
         h[2]->SetBinContent(iDest,c/bw);
         h[2]->SetBinError(iDest,e/bw);
      } else {
      }
   }
}

void CreateHistogramCopies(TH2 *h[3],TUnfoldBinning const *binningX) {
   h[1]=0;
   h[2]=0;
}

TH2 *AddOverflowXY(TH2 *h,double widthX,double widthY) {
   // add overflow bin to X-axis
   int nx=h->GetNbinsX();
   int ny=h->GetNbinsY();
   double *xBins=new double[nx+2];
   double *yBins=new double[ny+2];
   for(int i=1;i<=nx;i++) {
      xBins[i-1]=h->GetXaxis()->GetBinLowEdge(i);
   }
   xBins[nx]=h->GetXaxis()->GetBinUpEdge(nx);
   xBins[nx+1]=xBins[nx]+widthX;
   for(int i=1;i<=ny;i++) {
      yBins[i-1]=h->GetYaxis()->GetBinLowEdge(i);
   }
   yBins[ny]=h->GetYaxis()->GetBinUpEdge(ny);
   yBins[ny+1]=yBins[ny]+widthY;
   TString name(h->GetName());
   name+="U";
   TH2 *r=new TH2D(name,h->GetTitle(),nx+1,xBins,ny+1,yBins);
   for(int ix=0;ix<=nx+1;ix++) {
     for(int iy=0;iy<=ny+1;iy++) {
        r->SetBinContent(ix,iy,h->GetBinContent(ix,iy));
        r->SetBinError(ix,iy,h->GetBinError(ix,iy));
     }
   }
   delete [] yBins;
   delete [] xBins;
   return r;
}

TH1 *AddOverflowX(TH1 *h,double widthX) {
   // add overflow bin to X-axis
   int nx=h->GetNbinsX();
   double *xBins=new double[nx+2];
   for(int i=1;i<=nx;i++) {
      xBins[i-1]=h->GetXaxis()->GetBinLowEdge(i);
   }
   xBins[nx]=h->GetXaxis()->GetBinUpEdge(nx);
   xBins[nx+1]=xBins[nx]+widthX;
   TString name(h->GetName());
   name+="U";
   TH1 *r=new TH1D(name,h->GetTitle(),nx+1,xBins);
   for(int ix=0;ix<=nx+1;ix++) {
      r->SetBinContent(ix,h->GetBinContent(ix));
      r->SetBinError(ix,h->GetBinError(ix));
   }
   delete [] xBins;
   return r;
}

void DrawOverflowX(TH1 *h,double posy) {
  double x1=h->GetXaxis()->GetBinLowEdge(h->GetNbinsX());
  double x2=h->GetXaxis()->GetBinUpEdge(h->GetNbinsX());
  double y0=h->GetYaxis()->GetBinLowEdge(1);
  double y2=h->GetYaxis()->GetBinUpEdge(h->GetNbinsY());;
  if(h->GetDimension()==1) {
     y0=h->GetMinimum();
     y2=h->GetMaximum();
  }
  double w1=-0.3;
  TText *textX=new TText((1.+w1)*x2-w1*x1,(1.-posy)*y0+posy*y2,"Overflow bin");
  textX->SetNDC(kFALSE);
  textX->SetTextSize(0.05);
  textX->SetTextAngle(90.);
  textX->Draw();
  TLine *lineX=new TLine(x1,y0,x1,y2);
  lineX->Draw();
}

void DrawOverflowY(TH1 *h,double posx) {
   double x0=h->GetXaxis()->GetBinLowEdge(1);
  double x2=h->GetXaxis()->GetBinUpEdge(h->GetNbinsX());
  double y1=h->GetYaxis()->GetBinLowEdge(h->GetNbinsY());;
  double y2=h->GetYaxis()->GetBinUpEdge(h->GetNbinsY());;
  double w1=-0.3;
  TText *textY=new TText((1.-posx)*x0+posx*x2,(1.+w1)*y1-w1*y2,"Overflow bin");
  textY->SetNDC(kFALSE);
  textY->SetTextSize(0.05);
  textY->Draw();
  TLine *lineY=new TLine(x0,y1,x2,y1);
  lineY->Draw();
}

void DrawPadProbability(TH2 *h) {
  h->Draw("COLZ");
  h->SetTitle("migration probabilities;P_{T}(gen) [GeV];P_{T}(rec) [GeV]");
  DrawOverflowX(h,0.05);
  DrawOverflowY(h,0.35);
}

void DrawPadEfficiency(TH1 *h) {
  h->SetTitle("efficiency;P_{T}(gen) [GeV];#epsilon");
  h->SetLineColor(kBlue);
  h->SetMinimum(0.75);
  h->SetMaximum(1.0);
  h->Draw();
  DrawOverflowX(h,0.05);
  TLegend *legEfficiency=new TLegend(0.3,0.58,0.6,0.75);
  legEfficiency->SetBorderSize(0);
  legEfficiency->SetFillStyle(0);
  legEfficiency->SetTextSize(kLegendFontSize);
  legEfficiency->AddEntry(h,"reconstruction","l");
  legEfficiency->AddEntry((TObject*)0,"   efficiency","");
  legEfficiency->Draw();
}

void DrawPadReco(TH1 *histMcRec,TH1 *histMcbgrRec,TH1 *histDataRec,
                 TH1 *histDataUnfold,TH2 *histProbability,TH2 *histRhoij) {
   //gPad->SetLogy(kTRUE);
   double amax=0.0;
   for(int i=1;i<=histMcRec->GetNbinsX();i++) {
      amax=TMath::Max(amax,histMcRec->GetBinContent(i)
                      +5.0*histMcRec->GetBinError(i));
      amax=TMath::Max(amax,histDataRec->GetBinContent(i)
                      +2.0*histDataRec->GetBinError(i));
   }
   histMcRec->SetTitle("Reconstructed;P_{T}(rec);Nevent / GeV");
   histMcRec->Draw("HIST");
   histMcRec->SetLineColor(kBlue);
   histMcRec->SetMinimum(1.0);
   histMcRec->SetMaximum(amax);
   //histMcbgrRec->SetFillMode(1);
   histMcbgrRec->SetLineColor(kBlue-6);
   histMcbgrRec->SetFillColor(kBlue-10);
   histMcbgrRec->Draw("SAME HIST");

   TH1 * histFoldBack=0;
   if(histDataUnfold && histProbability && histRhoij) {
      histFoldBack=(TH1 *)
         histMcRec->Clone(histDataUnfold->GetName()+TString("_folded"));
      int nrec=histFoldBack->GetNbinsX();
      if((nrec==histProbability->GetNbinsY())&&
         (nrec==histMcbgrRec->GetNbinsX())&&
         (nrec==histDataRec->GetNbinsX())
         ) {
         for(int ix=1;ix<=nrec;ix++) {
            double sum=0.0;
            double sume2=0.0;
            for(int iy=0;iy<=histProbability->GetNbinsX()+1;iy++) {
               sum += histDataUnfold->GetBinContent(iy)*
                  histDataUnfold->GetBinWidth(iy)*
                  histProbability->GetBinContent(iy,ix);
               for(int iy2=0;iy2<=histProbability->GetNbinsX()+1;iy2++) {
                  sume2 += histDataUnfold->GetBinError(iy)*
                     histDataUnfold->GetBinWidth(iy)*
                     histProbability->GetBinContent(iy,ix)*
                     histDataUnfold->GetBinError(iy2)*
                     histDataUnfold->GetBinWidth(iy2)*
                     histProbability->GetBinContent(iy2,ix)*
                     histRhoij->GetBinContent(iy,iy2);
               }
            }
            sum /= histFoldBack->GetBinWidth(ix);
            sum += histMcbgrRec->GetBinContent(ix);
            histFoldBack->SetBinContent(ix,sum);
            histFoldBack->SetBinError(ix,TMath::Sqrt(sume2)
                                      /histFoldBack->GetBinWidth(ix));
         }
      } else {
         cout<<"can not fold back: "<<nrec
             <<" "<<histProbability->GetNbinsY()
             <<" "<<histMcbgrRec->GetNbinsX()
             <<" "<<histDataRec->GetNbinsX()
             <<"\n";
         exit(0);
      }

      histFoldBack->SetLineColor(kBlack);
      histFoldBack->SetMarkerStyle(0);
      histFoldBack->Draw("SAME HIST");
   }

   histDataRec->SetLineColor(kRed);
   histDataRec->SetMarkerColor(kRed);
   histDataRec->Draw("SAME");
   DrawOverflowX(histMcRec,0.5);

   TLegend *legRec=new TLegend(0.4,0.5,0.68,0.85);
   legRec->SetBorderSize(0);
   legRec->SetFillStyle(0);
   legRec->SetTextSize(kLegendFontSize);
   legRec->AddEntry(histMcRec,"MC total","l");
   legRec->AddEntry(histMcbgrRec,"background","f");
   if(histFoldBack) {
      int ndf=-kNbinC;
      double sumD=0.,sumF=0.,chi2=0.;
      for(int i=1;i<=histDataRec->GetNbinsX();i++) {
         //cout<<histDataRec->GetBinContent(i)<<" "<<histFoldBack->GetBinContent(i)<<" "<<" w="<<histFoldBack->GetBinWidth(i)<<"\n";
         sumD+=histDataRec->GetBinContent(i)*histDataRec->GetBinWidth(i);
         sumF+=histFoldBack->GetBinContent(i)*histFoldBack->GetBinWidth(i);
         double pull=(histFoldBack->GetBinContent(i)-histDataRec->GetBinContent(i))/histDataRec->GetBinError(i);
         chi2+= pull*pull;
         ndf+=1;
      }
      legRec->AddEntry(histDataRec,TString::Format("data N_{evt}=%.0f",sumD),"lp");
      legRec->AddEntry(histFoldBack,TString::Format("folded N_{evt}=%.0f",sumF),"l");
      legRec->AddEntry((TObject*)0,TString::Format("#chi^{2}=%.1f ndf=%d",chi2,ndf),"");
      //exit(0);
   } else {
      legRec->AddEntry(histDataRec,"data","lp");
   }
   legRec->Draw();
}

void DrawPadTruth(TH1 *histMcsigGen,TH1 *histDataGen,TH1 *histDataUnfold,
                  char const *text,double tau,vector<double> const *r,
                  TF1 *f) {
  //gPad->SetLogy(kTRUE);
   double amin=0.;
   double amax=0.;
   for(int i=1;i<=histMcsigGen->GetNbinsX();i++) {
      if(histDataUnfold) {
         amin=TMath::Min(amin,histDataUnfold->GetBinContent(i)
                         -1.1*histDataUnfold->GetBinError(i));
         amax=TMath::Max(amax,histDataUnfold->GetBinContent(i)
                         +1.1*histDataUnfold->GetBinError(i));
      }
      amin=TMath::Min(amin,histMcsigGen->GetBinContent(i)
                      -histMcsigGen->GetBinError(i));
      amin=TMath::Min(amin,histDataGen->GetBinContent(i)
                      -histDataGen->GetBinError(i));
      amax=TMath::Max(amax,histMcsigGen->GetBinContent(i)
                      +10.*histMcsigGen->GetBinError(i));
      amax=TMath::Max(amax,histDataGen->GetBinContent(i)
                      +2.*histDataGen->GetBinError(i));
   }
   histMcsigGen->SetMinimum(amin);
   histMcsigGen->SetMaximum(amax);

   histMcsigGen->SetTitle("Truth;P_{T};Nevent / GeV");
   histMcsigGen->SetLineColor(kBlue);
   histMcsigGen->Draw("HIST");
   histDataGen->SetLineColor(kRed);
   histDataGen->SetMarkerColor(kRed);
   histDataGen->SetMarkerSize(1.0);
   histDataGen->Draw("SAME HIST");
   if(histDataUnfold) {
      histDataUnfold->SetMarkerStyle(21);
      histDataUnfold->SetMarkerSize(0.7);
      histDataUnfold->Draw("SAME");
   }
   DrawOverflowX(histMcsigGen,0.5);

   if(f) {
      f->SetLineStyle(1);
      f->Draw("SAME");
   }

   TLegend *legTruth=new TLegend(0.32,0.65,0.6,0.9);
   legTruth->SetBorderSize(0);
   legTruth->SetFillStyle(0);
   legTruth->SetTextSize(kLegendFontSize);
   legTruth->AddEntry(histMcsigGen,"MC","l");
   if(!histDataUnfold) legTruth->AddEntry((TObject *)0,"  Landau(5,2)","");
   legTruth->AddEntry(histDataGen,"data","l");
   if(!histDataUnfold) legTruth->AddEntry((TObject *)0,"  Landau(6,1.8)","");
   if(histDataUnfold) {
      TString t;
      if(text) t=text;
      else t=histDataUnfold->GetName();
      if(tau>0) {
         t+=TString::Format(" #tau=%.2g",tau);
      }
      legTruth->AddEntry(histDataUnfold,t,"lp");
      if(r) {
         legTruth->AddEntry((TObject *)0,"test wrt data:","");
         legTruth->AddEntry((TObject *)0,TString::Format
                            ("#chi^{2}/%d=%.1f prob=%.3f",
                             (int)(*r)[3],(*r)[2]/(*r)[3],
                             TMath::Prob((*r)[2],(*r)[3])),"");
      }
   }
   if(f) {
      legTruth->AddEntry(f,"fit","l");
   }
   legTruth->Draw();
   if(histDataUnfold ) {
      TPad *subpad = new TPad("subpad","",0.35,0.29,0.88,0.68);
      subpad->SetFillStyle(0);
      subpad->Draw();
      subpad->cd();
      amin=0.;
      amax=0.;
      int istart=11;
      for(int i=istart;i<=histMcsigGen->GetNbinsX();i++) {
         amin=TMath::Min(amin,histMcsigGen->GetBinContent(i)
                         -histMcsigGen->GetBinError(i));
         amin=TMath::Min(amin,histDataGen->GetBinContent(i)
                         -histDataGen->GetBinError(i));
         amin=TMath::Min(amin,histDataUnfold->GetBinContent(i)
                         -histDataUnfold->GetBinError(i));
         amax=TMath::Max(amax,histMcsigGen->GetBinContent(i)
                         +histMcsigGen->GetBinError(i));
         amax=TMath::Max(amax,histDataGen->GetBinContent(i)
                         +histDataGen->GetBinError(i));
         amax=TMath::Max(amax,histDataUnfold->GetBinContent(i)
                         +histDataUnfold->GetBinError(i));
      }
      TH1 *copyMcsigGen=(TH1*)histMcsigGen->Clone();
      TH1 *copyDataGen=(TH1*)histDataGen->Clone();
      TH1 *copyDataUnfold=(TH1*)histDataUnfold->Clone();
      copyMcsigGen->GetXaxis()->SetRangeUser
         (copyMcsigGen->GetXaxis()->GetBinLowEdge(istart),
          copyMcsigGen->GetXaxis()->GetBinUpEdge(copyMcsigGen->GetNbinsX()-1));
      copyMcsigGen->SetTitle(";;");
      copyMcsigGen->GetYaxis()->SetRangeUser(amin,amax);
      copyMcsigGen->Draw("HIST");
      copyDataGen->Draw("SAME HIST");
      copyDataUnfold->Draw("SAME");
      if(f) {
         ((TF1 *)f->Clone())->Draw("SAME");
      }
   }
}

void DrawPadCorrelations(TH2 *h,
                         vector<pair<TF1*,vector<double> > > const *table) {
   h->SetMinimum(-1.);
   h->SetMaximum(1.);
   h->SetTitle("correlation coefficients;P_{T}(gen) [GeV];P_{T}(gen) [GeV]");
   h->Draw("COLZ");
   DrawOverflowX(h,0.05);
   DrawOverflowY(h,0.05);
   if(table) {
      TLegend *legGCor=new TLegend(0.13,0.6,0.5,0.8);
      legGCor->SetBorderSize(0);
      legGCor->SetFillStyle(0);
      legGCor->SetTextSize(kLegendFontSize);
      vector<double> const &r=(*table)[table->size()-1].second;
      legGCor->AddEntry((TObject *)0,TString::Format("min(#rho_{ij})=%5.2f",r[6]),"");
      legGCor->AddEntry((TObject *)0,TString::Format("max(#rho_{ij})=%5.2f",r[7]),"");
      legGCor->AddEntry((TObject *)0,TString::Format("avg(#rho_i)=%5.2f",r[5]),"");
      legGCor->Draw();
   }
}

TH1 *g_fcnHist=0;
TMatrixD *g_fcnMatrix=0;

void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag) {
  if(flag==0) {
    cout<<"fcn flag=0: npar="<<npar<<" gin="<<gin<<" par=[";
    for(int i=0;i<npar;i++) {
      cout<<" "<<u[i];
    }
    cout<<"]\n";
  }
   int n=g_fcnMatrix->GetNrows();
   TVectorD dy(n);
   double x0=0,y0=0.;
   for(int i=0;i<=n;i++) {
      double x1;
      if(i<1) x1=g_fcnHist->GetXaxis()->GetBinLowEdge(i+1);
      else x1=g_fcnHist->GetXaxis()->GetBinUpEdge(i);
      double y1=TMath::LandauI((x1-u[1])/u[2]);
      if(i>0) {
         double iy=u[0]*u[2]*(y1-y0)/(x1-x0);
         dy(i-1)=iy-g_fcnHist->GetBinContent(i);
         //cout<<"i="<<i<<" iy="<<iy<<" delta="<< dy(i-1)<<"\n";
      }
      x0=x1;
      y0=y1;
      //cout<<"i="<<i<<" y1="<<y1<<" x1="<<x1<<"\n";
   }
   TVectorD Hdy=(*g_fcnMatrix) * dy;
   //Hdy.Print();
   f=Hdy*dy;
   //exit(0);
}


TFitResultPtr DoFit(TH1 *h,TH2 *rho,TH1 *truth,const char *text,
                    vector<pair<TF1 *,vector<double> > > &table,int niter) {
   TString option="IESN";
   cout<<h->GetName()<<"\n";
   double gcorAvg=0.;
   double rhoMin=0.;
   double rhoMax=0.;
   if(rho) {
      g_fcnHist=h;
      int n=h->GetNbinsX()-1; // overflow is included as extra bin, exclude in fit
      TMatrixDSym v(n);
      //g_fcnMatrix=new TMatrixD(n,n);
      for(int i=0;i<n;i++) {
         for(int j=0;j<n;j++) {
            v(i,j)=rho->GetBinContent(i+1,j+1)*
               (h->GetBinError(i+1)*h->GetBinError(j+1));
         }
      }
      TMatrixDSymEigen ev(v);
      TMatrixD d(n,n);
      TVectorD di(ev.GetEigenValues());
      for(int i=0;i<n;i++) {
         if(di(i)>0.0) {
            d(i,i)=1./di(i);
         } else {
            cout<<"bad eigenvalue i="<<i<<" di="<<di(i)<<"\n";
            exit(0);
         }
      }
      TMatrixD O(ev.GetEigenVectors());
      TMatrixD DOT(d,TMatrixD::kMultTranspose,O);
      g_fcnMatrix=new TMatrixD(O,TMatrixD::kMult,DOT);
      TMatrixD test(*g_fcnMatrix,TMatrixD::kMult,v);
      int error=0;
      for(int i=0;i<n;i++) {
         if(TMath::Abs(test(i,i)-1.0)>1.E-7) {
            error++;
         }
         for(int j=0;j<n;j++) {
            if(i==j) continue;
            if(TMath::Abs(test(i,j)>1.E-7)) error++;
         }
      }
      // calculate global correlation coefficient (all bins)
      TMatrixDSym v1(n+1);
      rhoMin=1.;
      rhoMax=-1.;
      for(int i=0;i<=n;i++) {
         for(int j=0;j<=n;j++) {
            double rho_ij=rho->GetBinContent(i+1,j+1);
            v1(i,j)=rho_ij*
               (h->GetBinError(i+1)*h->GetBinError(j+1));
            if(i!=j) {
               if(rho_ij<rhoMin) rhoMin=rho_ij;
               if(rho_ij>rhoMax) rhoMax=rho_ij;
            }
         }
      }
      TMatrixDSymEigen ev1(v1);
      TMatrixD d1(n+1,n+1);
      TVectorD di1(ev1.GetEigenValues());
      for(int i=0;i<=n;i++) {
         if(di1(i)>0.0) {
            d1(i,i)=1./di1(i);
         } else {
            cout<<"bad eigenvalue i="<<i<<" di1="<<di1(i)<<"\n";
            exit(0);
         }
      }
      TMatrixD O1(ev1.GetEigenVectors());
      TMatrixD DOT1(d1,TMatrixD::kMultTranspose,O1);
      TMatrixD vinv1(O1,TMatrixD::kMult,DOT1);
      for(int i=0;i<=n;i++) {
         double gcor2=1.-1./(vinv1(i,i)*v1(i,i));
         if(gcor2>=0.0) {
            double gcor=TMath::Sqrt(gcor2);
            gcorAvg += gcor;
         } else {
            cout<<"bad global correlation "<<i<<" "<<gcor2<<"\n";
         }
      }
      gcorAvg /=(n+1);
      /* if(error) {
         v.Print();
         g_fcnMatrix->Print();
         exit(0);
         } */
      //g_fcnMatrix->Invert();
      //from: HFitImpl.cxx
      // TVirtualFitter::FCNFunc_t  userFcn = 0;
      //    typedef void   (* FCNFunc_t )(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
      // userFcn = (TVirtualFitter::GetFitter())->GetFCN();
      //  (TVirtualFitter::GetFitter())->SetUserFunc(f1);
      //...
      //fitok = fitter->FitFCN( userFcn );

      TVirtualFitter::Fitter(h)->SetFCN(fcn);
      option += "U";

   }
   double xmax=h->GetXaxis()->GetBinUpEdge(h->GetNbinsX()-1);
   TF1 *landau=new TF1(text,"[0]*TMath::Landau(x,[1],[2],0)",
                       0.,xmax);
   landau->SetParameter(0,6000.);
   landau->SetParameter(1,5.);
   landau->SetParameter(2,2.);
   landau->SetParError(0,10.);
   landau->SetParError(1,0.5);
   landau->SetParError(2,0.1);
   TFitResultPtr s=h->Fit(landau,option,0,0.,xmax);
   vector<double> r(8);
   int np=landau->GetNpar();
   fcn(np,0,r[0],landau->GetParameters(),0);
   r[1]=h->GetNbinsX()-1-landau->GetNpar();
   for(int i=0;i<h->GetNbinsX()-1;i++) {
      double di=h->GetBinContent(i+1)-truth->GetBinContent(i+1);
      if(g_fcnMatrix) {
         for(int j=0;j<h->GetNbinsX()-1;j++) {
            double dj=h->GetBinContent(j+1)-truth->GetBinContent(j+1);
            r[2]+=di*dj*(*g_fcnMatrix)(i,j);
         }
      } else {
         double pull=di/h->GetBinError(i+1);
         r[2]+=pull*pull;
      }
      r[3]+=1.0;
   }
   r[4]=niter;
   if(!niter) r[4]=0.25;
   r[5]=gcorAvg;
   r[6]=rhoMin;
   r[7]=rhoMax;
   if(rho) {
      g_fcnHist=0;
      delete g_fcnMatrix;
      g_fcnMatrix=0;
   }
   table.push_back(make_pair(landau,r));
   return s;
}

#ifdef WITH_IDS

//===================== interface to IDS unfolding code follows here
// contact Bogdan Malescu to find it

#include "ids_code.cc"

void IDSfirst(TVectorD *data, TVectorD *dataErr, TMatrixD *A_, Double_t lambdaL_, TVectorD* &unfres1IDS_,TVectorD *&soustr){

   int N_=data->GetNrows();
   soustr = new TVectorD(N_);
   for( Int_t i=0; i<N_; i++ ){ (*soustr)[i] = 0.; }
   unfres1IDS_ = Unfold( data, dataErr, A_, N_, lambdaL_, soustr );
}

void IDSiterate(TVectorD *data, TVectorD *dataErr, TMatrixD *A_, TMatrixD *Am_, Double_t lambdaU_, Double_t lambdaM_, Double_t lambdaS_,TVectorD* &unfres2IDS_ ,TVectorD *&soustr) {

   int N_=data->GetNrows();
   ModifyMatrix( Am_, A_, unfres2IDS_, dataErr, N_, lambdaM_, soustr, lambdaS_ );
   delete unfres2IDS_;
   unfres2IDS_ = Unfold( data, dataErr, Am_, N_, lambdaU_, soustr );
}

#endif
