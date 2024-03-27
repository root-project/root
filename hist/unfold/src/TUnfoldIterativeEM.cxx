#include "TUnfoldIterativeEM.h"
#include "TUnfoldBinning.h"
#include <TVectorD.h>

ClassImp(TUnfoldIterativeEM)

TUnfoldIterativeEM::TUnfoldIterativeEM(void) {
   f_inputBins=nullptr;
   f_outputBins=nullptr;
   f_constInputBins=nullptr;
   f_constOutputBins=nullptr;
   fA=nullptr;
   fEpsilon=nullptr;
   fX0=nullptr;
   fY=nullptr;
   fBgr=nullptr;
   fX=nullptr;
   fDXDY=nullptr;
}

TUnfoldIterativeEM::TUnfoldIterativeEM
(const TH2 *hist_A, TUnfold::EHistMap histmap,
 const TUnfoldBinning *outputBins,const TUnfoldBinning *inputBins) {
   // copied from TUnfoldDensity
   TAxis const *genAxis,*detAxis;
   if(histmap==TUnfold::kHistMapOutputHoriz) {
      genAxis=hist_A->GetXaxis();
      detAxis=hist_A->GetYaxis();
   } else {
      genAxis=hist_A->GetYaxis();
      detAxis=hist_A->GetXaxis();
   }
   if(!inputBins) {
      f_inputBins=new TUnfoldBinning(*detAxis,0,0);
      f_constInputBins=f_inputBins;
   } else {
      f_inputBins=nullptr;
      f_constInputBins=inputBins;
   }
   if(!outputBins) {
      f_outputBins=new TUnfoldBinning(*genAxis,1,1);
      f_constOutputBins=f_outputBins;
   } else {
      f_outputBins=nullptr;
      f_constOutputBins=outputBins;
   }
   int nGen=f_constOutputBins->GetEndBin();
   int nRec=f_constInputBins->GetEndBin();
   fA=new TMatrixD(nRec-1,nGen);
   fEpsilon=new TVectorD(nGen);
   fX0=new TVectorD(nGen);
   for(int iGen=0;iGen<nGen;iGen++) {
      double sum=0.;
      for(int iRec=0;iRec<=nRec;iRec++) {
         double c;
         if(histmap==TUnfold::kHistMapOutputHoriz) {
            c= hist_A->GetBinContent(iGen,iRec);
         } else {
            c= hist_A->GetBinContent(iRec,iGen);
         }
         if((iRec>0)&&(iRec<=fA->GetNrows())) {
            (*fA)(iRec-1,iGen)=c;
         }
         sum +=c;
      }
      double epsilon=0.;
      if(sum!=0.) {
         for(int iRec=0;iRec<fA->GetNrows();iRec++) {
            (*fA)(iRec,iGen) /=sum;
            epsilon += (*fA)(iRec,iGen);
         }
      }
      (*fEpsilon)(iGen)=epsilon;
      (*fX0)(iGen)=sum;
   }
   fStep=-1;
   fScaleBias=1.;
   fY=new TVectorD(nRec-1);
   fBgr=new TVectorD(nRec-1);
   fX=new TVectorD(*fX0);
   fDXDY=new TMatrixD(nGen,nRec-1);
}

TUnfoldIterativeEM::~TUnfoldIterativeEM() {
   if(f_inputBins) delete f_inputBins;
   if(f_outputBins) delete f_outputBins;
   if(fA) delete fA;
   if(fEpsilon) delete fEpsilon;
   if(fX0) delete fX0;
   if(fY) delete fY;
   if(fBgr) delete fBgr;
   if(fX) delete fX;
   if(fDXDY) delete fDXDY;
}

void TUnfoldIterativeEM::DoUnfold(Int_t numIterations) {
   if(numIterations<fStep) {
      Reset();
   }
   while(fStep<numIterations) {
      IterateOnce();
   }
}

Int_t TUnfoldIterativeEM::SetInput(const TH1 *hist_y,Double_t scaleBias) {
   int nRec=f_constInputBins->GetEndBin();
   for(int iRec=1;iRec<nRec;iRec++) {
      (*fY)(iRec-1)=hist_y->GetBinContent(iRec);
   }
   // reset start value
   fScaleBias=scaleBias;
   Reset();
   return 0;
}

void TUnfoldIterativeEM::Reset(void) {
   for(int iGen=0;iGen<fX->GetNrows();iGen++) {
      (*fX)=fScaleBias*(*fX0);
   }
   for(int i=0;i<fDXDY->GetNrows();i++) {
      for(int j=0;j<fDXDY->GetNcols();j++) {
         (*fDXDY)(i,j)=0.;
      }
   }
   fStep=-1;
}

void TUnfoldIterativeEM::SubtractBackground(const TH1 *hist_bgr,const char * /*name*/,Double_t scale) {
   int nRec=f_constInputBins->GetEndBin();
   for(int iRec=1;iRec<nRec;iRec++) {
      (*fBgr)(iRec-1)+=hist_bgr->GetBinContent(iRec)*scale;
   }
}

void TUnfoldIterativeEM::DoUnfold
(Int_t nIter,const TH1 *hist_y, Double_t scaleBias) {
   SetInput(hist_y,scaleBias);
   DoUnfold(nIter);
}

void TUnfoldIterativeEM::IterateOnce(void) {
   TVectorD Ax_plus_bgr=(*fA)*(*fX)+(*fBgr);
   TMatrixD f(fY->GetNrows(),1);
   TMatrixD dfdy(fY->GetNrows(),fY->GetNrows());
   TMatrixD ADXDY(*fA,TMatrixD::kMult,*fDXDY);
   for(int i=0;i<f.GetNrows();i++) {
      f(i,0)=(*fY)(i)/Ax_plus_bgr(i);
      // dfdx(i,j)=-f(i,0)/Ax(i)*(*fA)(i,j);
      dfdy(i,i)=1./Ax_plus_bgr(i);
      for(int j=0;j<fY->GetNrows();j++) {
         dfdy(i,j) -= f(i,0)/Ax_plus_bgr(i)*ADXDY(i,j);
      }
   }
   //
   TMatrixD At_f(*fA,TMatrixD::kTransposeMult,f);
   TMatrixD At_dfdy(*fA,TMatrixD::kTransposeMult,dfdy);
   for(int i=0;i<fX->GetNrows();i++) {
      if((*fEpsilon)(i)<=0.) continue;
      double factor=At_f(i,0)/(*fEpsilon)(i);
      for(int j=0;j<fY->GetNrows();j++) {
         (*fDXDY)(i,j) *= factor;
         (*fDXDY)(i,j) += (*fX)(i)/(*fEpsilon)(i)*At_dfdy(i,j);
      }
      (*fX)(i) *= factor;
   }
   fStep++;
}

Int_t TUnfoldIterativeEM::ScanSURE(Int_t nIterMax,TGraph **graphSURE,
                                      TGraph **df_deviance) {
   Reset();
   double minSURE=GetSURE();
   int stepSURE=fStep;
   TVectorD X_SURE(*fX);
   TMatrixD DXDY_SURE(*fDXDY);
   std::vector<double> nIter,scanSURE,scanDeviance,scanDF;
   nIter.push_back(fStep);
   scanSURE.push_back(minSURE);
   scanDeviance.push_back(GetDeviance());
   scanDF.push_back(GetDF());
   Info("TUnfoldIterativeEM::ScanSURE",
        "step=%d SURE=%lf DF=%lf deviance=%lf",
        fStep,*scanSURE.rbegin(),*scanDF.rbegin(),*scanDeviance.rbegin());
   while(fStep<nIterMax) {
      DoUnfold(fStep+1);
      double SURE=GetSURE();
      nIter.push_back(fStep);
      scanSURE.push_back(SURE);
      scanDeviance.push_back(GetDeviance());
      scanDF.push_back(GetDF());
      Info("TUnfoldIterativeEM::ScanSURE",
           "step=%d SURE=%lf DF=%lf deviance=%lf",
           fStep,*scanSURE.rbegin(),*scanDF.rbegin(),*scanDeviance.rbegin());
      if(SURE<minSURE) {
         minSURE=SURE;
         X_SURE=*fX;
         DXDY_SURE=*fDXDY;
         stepSURE=fStep;
      }
   }
   if(graphSURE) {
      *graphSURE=new TGraph(nIter.size(),nIter.data(),scanSURE.data());
   }
   if(df_deviance) {
      *df_deviance=new TGraph
         (scanDeviance.size(),scanDF.data(),scanDeviance.data());
   }

   *fX=X_SURE;
   *fDXDY=DXDY_SURE;
   fStep=stepSURE;

   return fStep;
}

TH1 *TUnfoldIterativeEM::GetOutput
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *projectionMode,
 Bool_t useAxisBinning) const {
   TUnfoldBinning const *binning=f_constOutputBins->FindNode(distributionName);
   Int_t *binMap=nullptr;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,projectionMode);
   if(r) {
      for(Int_t i=0;i<binning->GetEndBin();i++) {
         Int_t destBin=binMap[i];
         if(destBin<0.) continue;
         r->SetBinContent(destBin,r->GetBinContent(destBin)+(*fX)(i));
         Double_t Vii=0.;
         for(Int_t k=0;k<binning->GetEndBin();k++) {
            if(binMap[k]!=destBin) continue;
            for(int j=0;j<fDXDY->GetNcols();j++) {
               // add up Poisson errors squared
               Vii += (*fDXDY)(i,j)*(*fY)(j)*(*fDXDY)(k,j);
            }
         }
         r->SetBinError(destBin,TMath::Sqrt(r->GetBinError(destBin)+Vii));
      }
   }
   if(binMap) {
     delete [] binMap;
   }
   return r;
}

TH1 *TUnfoldIterativeEM::GetFoldedOutput
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *projectionMode,
 Bool_t useAxisBinning,Bool_t addBgr) const {
   TUnfoldBinning const *binning=f_constInputBins->FindNode(distributionName);
   Int_t *binMap=nullptr;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,projectionMode);
   if(r) {
      TVectorD folded((*fA)*(*fX));
      if(addBgr) folded+= *fBgr;
      TMatrixD dFoldedDY((*fA)*(*fDXDY));
      for(Int_t i=1;i<binning->GetEndBin();i++) {
         Int_t destBin=binMap[i];
         if(destBin<0.) continue;
         r->SetBinContent(destBin,r->GetBinContent(destBin)+folded(i-1));
         Double_t Vii=0.;
         for(Int_t k=1;k<binning->GetEndBin();k++) {
            if(binMap[k]!=destBin) continue;
            for(int j=0;j<dFoldedDY.GetNcols();j++) {
               // add up Poisson errors squared
               Vii += dFoldedDY(i-1,j)*(*fY)(j)*dFoldedDY(k-1,j);
            }
         }
         r->SetBinError(destBin,TMath::Sqrt(r->GetBinError(destBin)+Vii));
      }
   }
   if(binMap) {
     delete [] binMap;
   }
   return r;
}


Double_t TUnfoldIterativeEM::GetDeviance(void) const {
   // fold data with matrix
   TVectorD Ax_plus_bgr=(*fA)*(*fX)+(*fBgr);
   double r=0.;
   for(int i=0;i<Ax_plus_bgr.GetNrows();i++) {
      double n=(*fY)(i);
      double mu=Ax_plus_bgr(i);
      if(n>0.) {
         r += 2.* (mu-n-n*TMath::Log(mu/n));
      } else if(mu>0.) {
         r += 2.*mu;
      }
   }
   return r;
}

Double_t TUnfoldIterativeEM::GetDF(void) const {
   double r=0.;
   for(int i=0;i<fA->GetNrows();i++) {
      for(int j=0;j<fA->GetNcols();j++) {
         r += (*fA)(i,j)*(*fDXDY)(j,i);
      }
   }
   return r;
}

Double_t TUnfoldIterativeEM::GetSURE(void) const {
   return GetDeviance()+2.*GetDF();
}
