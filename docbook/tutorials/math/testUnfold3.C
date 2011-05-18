// Test program for the class TUnfoldSys
// Author: Stefan Schmitt
// DESY, 14.10.2008

//  Version 16, parallel to changes in TUnfold
//
//  History:
//     Version 15, simple example including background subtraction

#include <TMath.h>
#include <TCanvas.h>
#include <TRandom3.h>
#include <TFitter.h>
#include <TF1.h>
#include <TStyle.h>
#include <TVector.h>
#include <TGraph.h>

#include <TUnfoldSys.h>

using namespace std;

///////////////////////////////////////////////////////////////////////
// 
// Test program for the class TUnfoldSys
//
//  the goal is to unfold the underlying "true" distribution of a variable Pt
//
//  the reconstructed Pt is measured in 24 bins from 4 to 28 
//  the generator-level Pt is unfolded into 10 bins from 6 to 26
//    plus underflow bin from 0 to 6
//    plus overflow bin above 26 
//  there are two background sources
//       bgr1 and bgr2
//  the signal has a finite trigger efficiency at a threshold of 8 GeV
//
//  two types of systematic error are studied:
//    SYS1: variation of the input shape
//    SYS2: variation of the trigger efficiency at threshold
//
//  Finally, the unfolding is compared to a "bin-by-bin" correction method
//
///////////////////////////////////////////////////////////////////////

TRandom *rnd=0;

Double_t GenerateEvent(const Double_t *parm,
                       const Double_t *triggerParm,
                       Double_t *intLumi,
                       Bool_t *triggerFlag,
                       Double_t *ptGen,Int_t *iType)
{
   // generate an event
   // input:
   //      parameters for the event generator
   // return value:
   //      reconstructed Pt
   // output to pointers:
   //      integrated luminosity
   //      several variables only accessible on generator level
   //
   // the parm array defines the physical parameters
   //  parm[0]: background source 1 fraction
   //  parm[1]: background source 2 fraction
   //  parm[2]: lower limit of generated Pt distribution
   //  parm[3]: upper limit of generated Pt distribution
   //  parm[4]: exponent for Pt distribution signal
   //  parm[5]: exponent for Pt distribution background 1
   //  parm[6]: exponent for Pt distribution background 2
   //  parm[7]: resolution parameter a goes with sqrt(Pt)
   //  parm[8]: resolution parameter b goes with Pt
   //  triggerParm[0]: trigger threshold turn-on position
   //  triggerParm[1]: trigger threshold turn-on width
   //  triggerParm[2]: trigger efficiency for high pt
   //
   // intLumi is advanced bu 1 for each *generated* event
   // for data, several events may be generated, until one passes the trigger
   //
   // some generator-level quantities are also returned:
   //   triggerFlag: whether the event passed the trigger threshold
   //   ptGen: the generated pt
   //   iType: which type of process was simulated
   //
   // the "triggered" also has another meaning:
   //   if(triggerFlag==0)   only triggered events are returned
   //
   // Usage to generate data events
   //      ptObs=GenerateEvent(parm,triggerParm,0,0,0)
   //
   // Usage to generate MC events
   //      ptGen=GenerateEvent(parm,triggerParm,&triggerFlag,&ptGen,&iType);
   //
   Double_t ptObs;
   Bool_t isTriggered=kFALSE;
   do {
      Int_t itype;
      Double_t ptgen;
      Double_t f=rnd->Rndm();
      // decide whether this is background or signal
      itype=0;
      if(f<parm[0]) itype=1;
      else if(f<parm[0]+parm[1]) itype=2;
      // generate Pt according to distribution  pow(ptgen,a)
      // get exponent
      Double_t a=parm[4+itype];
      Double_t a1=a+1.0;
      Double_t t=rnd->Rndm();
      if(a1 == 0.0) {
         Double_t x0=TMath::Log(parm[2]);
         ptgen=TMath::Exp(t*(TMath::Log(parm[3])-x0)+x0);
      } else {
         Double_t x0=pow(parm[2],a1);
         ptgen=pow(t*(pow(parm[3],a1)-x0)+x0,1./a1);
      }
      if(iType) *iType=itype;
      if(ptGen) *ptGen=ptgen;
      
      // smearing in Pt with large asymmetric tail
      Double_t sigma=
         TMath::Sqrt(parm[7]*parm[7]*ptgen+parm[8]*parm[8]*ptgen*ptgen);
      ptObs=rnd->BreitWigner(ptgen,sigma);

      // decide whether event was triggered
      Double_t triggerProb =
         triggerParm[2]/(1.+TMath::Exp((triggerParm[0]-ptObs)/triggerParm[1]));
      isTriggered= rnd->Rndm()<triggerProb;
      (*intLumi) ++;
   } while((!triggerFlag) && (!isTriggered));
   // return trigger decision
   if(triggerFlag) *triggerFlag=isTriggered;
   return ptObs;
}

//int main(int argc, char *argv[])
void testUnfold3()
{
  // switch on histogram errors
  TH1::SetDefaultSumw2();

  // show fit result
  gStyle->SetOptFit(1111);

  // random generator
  rnd=new TRandom3();

  // data and MC luminosities
  Double_t const lumiData= 30000;
  Double_t const lumiMC  =1000000;

  // Binning
  // reconstructed mass
  Int_t const nDet=24;
  Double_t const xminDet=4.0;
  Double_t const xmaxDet=28.0;
  // signal contributions
  Int_t const nGen=10;
  Double_t const xminGen= 6.0;
  Double_t const xmaxGen=26.0;

  //========================
  // Step 1: fill histograms

  // =================== data true parameters ============================
  //
  // in real life these are unknown
  //
  static Double_t parm_data[]={
     0.05, // fraction of background 1 (on generator level)
     0.05, // fraction of background 2 (on generator level)
     3.5, // lower Pt cut (generator level)
     100.,// upper Pt cut (generator level)
     -3.0,// signal exponent
     0.1, // background 1 exponent
     -0.5, // background 2 exponent
     0.2, // energy resolution a term
     0.01, // energy resolution b term
  };
  static Double_t triggerParm_data[]={8.0, // threshold is 8 GeV
                               4.0, // width is 4 GeV
                               0.95 // high Pt efficiency os 95%
  };

  //============================================
  // generate data distribution
  TH1D *histDetDATA=new TH1D("PT(data)",";Pt(obs)",nDet,xminDet,xmaxDet);
  // second data distribution, with generator-level bins
  // for bin-by-bin correction study
  TH1D *histDetDATAbbb=new TH1D("PT(data,bbb)",";Pt(obs,bbb)",
                                nGen,xminGen,xmaxGen);
  // in real life we do not have this distribution...
  TH1D *histGenDATA=new TH1D("PT(MC,signal,gen)",";Pt(gen)",
                             nGen,xminGen,xmaxGen);

  Double_t intLumi=0.0;
  while(intLumi<lumiData) {
     Int_t iTypeGen;
     Bool_t isTriggered;
     Double_t ptGen;
     Double_t ptObs=GenerateEvent(parm_data,triggerParm_data,&intLumi,
                                  &isTriggered,&ptGen,&iTypeGen);
     if(isTriggered) {
        histDetDATA->Fill(ptObs);
        histDetDATAbbb->Fill(ptObs);
     }
     // fill generator-level signal histogram for data
     // for real data, this does not exist
     if(iTypeGen==0) histGenDATA->Fill(ptGen);
  }


  // =================== MC parameters ============================
  // default settings
  static Double_t parm_MC[]={
     0.05, // fraction of background 1 (on generator level)
     0.05, // fraction of background 2 (on generator level)
     3.5, // lower Pt cut (generator level)
     100.,// upper Pt cut (generator level)
     -4.0,// signal exponent !!! UNKNOWN !!! steeper than in data
          //                                 to illustrate bin-by-bin bias
     0.1, // background 1 exponent
     -0.5, // background 2 exponent
     0.2, // energy resolution a term
     0.01, // energy resolution b term
  };
  static Double_t triggerParm_MC[]={8.0, // threshold is 8 GeV
                             4.0, // width is 4 GeV
                             0.95 // high Pt efficiency os 95%
  };

  
  // study trigger systematics: change parameters for trigger threshold
  static Double_t triggerParm_MCSYS1[]={7.7, // threshold is 7 GeV
                                        3.7, // width is 3 GeV
                                        0.95 // high Pt efficiency is 9%
  };
  // study dependence on initial signal shape
  static Double_t parm_MC_SYS2[]={
     0.0, // fraction of background: not needed
     0.0, // fraction of background: not needed
     3.5, // lower Pt cut (generator level)
     100.,// upper Pt cut (generator level)
     -2.0, // signal exponent changed
     0.1, // background 1 exponent
     -0.5, // background 2 exponent
     0.2, // energy resolution a term
     0.01, // energy resolution b term
  };

  //============================================
  // generate MC distributions
  // detector-level histograms
  TH1D *histDetMC[3];
  histDetMC[0]=new TH1D("PT(MC,signal)",";Pt(obs)",nDet,xminDet,xmaxDet);
  histDetMC[1]=new TH1D("PT(MC,bgr1)",";Pt(obs)",nDet,xminDet,xmaxDet);
  histDetMC[2]=new TH1D("PT(MC,bgr2)",";Pt(obs)",nDet,xminDet,xmaxDet);
  TH1D *histDetMCall=new TH1D("PT(MC,all)",";Pt(obs)",nDet,xminDet,xmaxDet);
  TH1D *histDetMCbgr=new TH1D("PT(MC,bgr)",";Pt(obs)",nDet,xminDet,xmaxDet);
  // another set of detector level histograms, this time with
  // generator-level binning, to try the bin-by-bin correction
  TH1D *histDetMCbbb[3];
  histDetMCbbb[0]=new TH1D("PT(MC,sig)",";Pt(obs,bbb)",nGen,xminGen,xmaxGen);
  histDetMCbbb[1]=new TH1D("PT(MC,bgr1)bbb",";Pt(obs,bbb)",nGen,xminGen,xmaxGen);
  histDetMCbbb[2]=new TH1D("PT(MC,bgr2)bbb",";Pt(obs,bbb)",nGen,xminGen,xmaxGen);
  // true signal distribution, two different binnings
  TH1D *histGenMC=new TH1D("PT(MC,signal,gen)bbb",";Pt(gen)",
                         nGen,xminGen,xmaxGen);
  TH2D *histGenDet=new TH2D("PT(MC,signal,gen,det)",";Pt(gen);Pt(det)",
                            nGen,xminGen,xmaxGen,nDet,xminDet,xmaxDet);

  // weighting factor to accomodate for the different luminosity in data and MC
  Double_t lumiWeight = lumiData/lumiMC;
  intLumi=0.0;
  while(intLumi<lumiMC) {
     Int_t iTypeGen;
     Bool_t isTriggered;
     Double_t ptGen;
     Double_t ptObs=GenerateEvent(parm_MC,triggerParm_MC,&intLumi,&isTriggered,
                                  &ptGen,&iTypeGen);
     // detector-level distributions
     // require trigger
     if(isTriggered) {
        // fill signal, bgr1,bgr2 into separate histograms
        histDetMC[iTypeGen]->Fill(ptObs,lumiWeight);
        histDetMCbbb[iTypeGen]->Fill(ptObs,lumiWeight);
        // fill total MC prediction (signal+bgr1+bgr2)
        histDetMCall->Fill(ptObs,lumiWeight);
        // fill bgr only MC prediction (bgr1+bgr2)
        if(iTypeGen>0) {
           histDetMCbgr->Fill(ptObs,lumiWeight);
        }
     }
     // generator level distributions (signal only)
     if(iTypeGen==0) {
        histGenMC->Fill(ptGen,lumiWeight);
     }
     // matrix dectribing the migrations (signal only)
     if(iTypeGen==0) {
        // if event was triggered, fill histogram with ptObs
        if(isTriggered) {
           histGenDet->Fill(ptGen,ptObs,lumiWeight);
        } else {
           // not triggered: fill detector-level underflow bin
           histGenDet->Fill(ptGen,-999.,lumiWeight);
        }
     }
  }

  // generate another MC, this time with changed trigger settings
  // fill 2D histogram and histograms for bin-by-bin study
  TH2D *histGenDetSYS1=new TH2D("PT(MC,signal,gen,det,sys1)",
                                ";Pt(gen);Pt(obs)",
                                nGen,xminGen,xmaxGen,nDet,xminDet,xmaxDet);
  TH1D *histGenSYS1=new TH1D("PT(MC,signal,gen,sys1)",
                             ";Pt(obs)",
                             nGen,xminGen,xmaxGen);
  TH1D *histDetSYS1bbb=new TH1D("PT(MC,signal,det,sys1,bbb)",
                                ";Pt(obs)",
                                nGen,xminGen,xmaxGen);
  intLumi=0.;
  while(intLumi<lumiMC) {
     Int_t iTypeGen;
     Bool_t isTriggered;
     Double_t ptGen;
     Double_t ptObs=GenerateEvent(parm_MC,triggerParm_MCSYS1,&intLumi,
                                  &isTriggered,&ptGen,&iTypeGen);
     // matrix describing the migrations (signal only)
     if(iTypeGen==0) {
        // if event was triggered, fill histogram with ptObs
        if(isTriggered) {
           histGenDetSYS1->Fill(ptGen,ptObs,lumiWeight);
        } else {
           // not triggered: fill detector-level underflow bin
           histGenDetSYS1->Fill(ptGen,-999.,lumiWeight);
        }
     }
     // generator level distribution for bin-by-bin study
     if(iTypeGen==0) {
        histGenSYS1->Fill(ptGen,lumiWeight);
        // detector level for bin-by-bin study
        if(isTriggered) {
           histDetSYS1bbb->Fill(ptObs,lumiWeight);
        }
     }
  }
  // generate a third MC, this time with changed initial distribution
  // only the 2D histogram is filled
  TH2D *histGenDetSYS2=new TH2D("PT(MC,signal,gen,det,sys2)",
                                ";Pt(gen);Pt(det)",
                                nGen,xminGen,xmaxGen,nDet,xminDet,xmaxDet);
  TH1D *histGenSYS2=new TH1D("PT(MC,signal,gen,sys2)",
                             ";Pt(obs)",
                             nGen,xminGen,xmaxGen);
  TH1D *histDetSYS2bbb=new TH1D("PT(MC,signal,det,sys2,bbb)",
                                ";Pt(obs)",
                                nGen,xminGen,xmaxGen);
  intLumi=0.0;
  while(intLumi<lumiMC) {
     Int_t iTypeGen;
     Bool_t isTriggered;
     Double_t ptGen;
     Double_t ptObs=GenerateEvent(parm_MC_SYS2,triggerParm_MC,&intLumi,
                                  &isTriggered,&ptGen,&iTypeGen);
     // matrix dectribing the migrations (signal only)
     if(iTypeGen==0) {
        // if event was triggered, fill histogram with ptObs
        if(isTriggered) {
           histGenDetSYS2->Fill(ptGen,ptObs,lumiWeight);
        } else {
           // not triggered: fill detector-level underflow bin
           histGenDetSYS2->Fill(ptGen,-999.,lumiWeight);
        }
     }
     if(iTypeGen==0) {
        // generator level distribution for bin-by-bin study
        histGenSYS2->Fill(ptGen,lumiWeight);
        // detector level for bin-by-bin study
        if(isTriggered) {
           histDetSYS2bbb->Fill(ptObs,lumiWeight);
        }
     }
  }

  //========================
  // Step 2: unfolding
  //
  // this is based on a matrix (TH2D) which describes the connection
  // between
  //   nDet    Detector bins
  //   nGen+2  Generator level bins
  //
  // why are there nGen+2 Generator level bins?
  //   the two extra bins are the underflow and overflow bin
  //   These are unfolded as well. Such bins are needed to
  //   accomodate for migrations from outside the phase-space into the
  //   observed phase-space.
  //
  // In addition there are overflow/underflow bins
  //   for the reconstructed variable. These are used to count events
  //   which are -NOT- reconstructed. Either because they migrate
  //   out of the reconstructed phase-space. Or because there are detector
  //   inefficiencies.
  //
  TUnfoldSys unfold(histGenDet,TUnfold::kHistMapOutputHoriz,
                    TUnfold::kRegModeSize,TUnfold::kEConstraintNone);

  // define the input vector (the measured data distribution)
  unfold.SetInput(histDetDATA);

  // subtract background, normalized to data luminosity
  //  with 10% scale error each
  Double_t scale_bgr=1.0;
  Double_t dscale_bgr=0.2;
  unfold.SubtractBackground(histDetMC[1],"background1",scale_bgr,dscale_bgr);
  unfold.SubtractBackground(histDetMC[2],"background2",scale_bgr,dscale_bgr);

  // add systematic errors
  // trigger threshold
  unfold.AddSysError(histGenDetSYS1,"trigger_SYS1",
                     TUnfold::kHistMapOutputHoriz,
                     TUnfoldSys::kSysErrModeMatrix);
  // calorimeter response
  unfold.AddSysError(histGenDetSYS2,"signalshape_SYS2",
                     TUnfold::kHistMapOutputHoriz,
                     TUnfoldSys::kSysErrModeMatrix);

  // run the unfolding
  Int_t nScan=30;
  TSpline *logTauX,*logTauY;
  TGraph *lCurve;
  // this method scans the parameter tau and finds the kink in the L curve
  // finally, the unfolding is done for the best choice of tau
  Int_t iBest=unfold.ScanLcurve(nScan,0.,0.,&lCurve,&logTauX,&logTauY);

  // save graphs with one point to visualize best choice of tau
  Double_t t[1],x[1],y[1];
  logTauX->GetKnot(iBest,t[0],x[0]);
  logTauY->GetKnot(iBest,t[0],y[0]);
  TGraph *bestLcurve=new TGraph(1,x,y);
  TGraph *bestLogTauLogChi2=new TGraph(1,t,x);

  // get unfolding output
  // Reminder: there are
  //   nGen+2  Generator level bins
  // But we are not interested in the additional bins.
  // The mapping is required to get the proper errors and correlations

  // Here a mapping is defined from the nGen+2 bins to nGen bins.
  Int_t *binMap=new Int_t[nGen+2];
  binMap[0] = -1;    // underflow bin should be ignored
  binMap[nGen+1]=-1; // overflow bin should be ignored
  for(Int_t i=1;i<=nGen;i++) {
     binMap[i]=i;  // map inner bins to output histogram bins
  }

  // this returns the unfolding output
  // The errors included at this point are:
  //    * statistical errros
  //    * background subtraction errors
  TH1D *histUnfoldStatBgr=new TH1D("PT(unfold,signal,stat+bgr)",";Pt(gen)",
                            nGen,xminGen,xmaxGen);
  unfold.GetOutput(histUnfoldStatBgr,binMap);

  // retreive error matrix of statistical errors only
  TH2D *histEmatStat=new TH2D("ErrMat(stat)",";Pt(gen);Pt(gen)",
                              nGen,xminGen,xmaxGen,nGen,xminGen,xmaxGen);
  unfold.GetEmatrixInput(histEmatStat,binMap);

  // retreive full error matrix
  // This includes also all systematic errors defined above
  TH2D *histEmatTotal=new TH2D("ErrMat(total)",";Pt(gen);Pt(gen)",
                               nGen,xminGen,xmaxGen,nGen,xminGen,xmaxGen);
  unfold.GetEmatrixTotal(histEmatTotal,binMap);

  // create two copies of the unfolded data, one with statistical errors
  // the other with total errors
  TH1D *histUnfoldStat=new TH1D("PT(unfold,signal,stat)",";Pt(gen)",
                                nGen,xminGen,xmaxGen);
  TH1D *histUnfoldTotal=new TH1D("PT(unfold,signal,total)",";Pt(gen)",
                                 nGen,xminGen,xmaxGen);
  for(Int_t i=0;i<nGen+2;i++) {
     Double_t c=histUnfoldStatBgr->GetBinContent(i);

     // histogram with unfolded data and stat errors
     histUnfoldStat->SetBinContent(i,c);
     histUnfoldStat->SetBinError
        (i,TMath::Sqrt(histEmatStat->GetBinContent(i,i)));

     // histogram with unfolded data and total errors
     histUnfoldTotal->SetBinContent(i,c); 
     histUnfoldTotal->SetBinError
        (i,TMath::Sqrt(histEmatTotal->GetBinContent(i,i)));
  }

  // create histogram with correlation matrix
  TH2D *histCorr=new TH2D("Corr(total)",";Pt(gen);Pt(gen)",
                          nGen,xminGen,xmaxGen,nGen,xminGen,xmaxGen);
  for(Int_t i=0;i<nGen+2;i++) {
     Double_t ei,ej;
     ei=TMath::Sqrt(histEmatTotal->GetBinContent(i,i));
     if(ei<=0.0) continue;
     for(Int_t j=0;j<nGen+2;j++) {
        ej=TMath::Sqrt(histEmatTotal->GetBinContent(j,j));
        if(ej<=0.0) continue;
        histCorr->SetBinContent(i,j,histEmatTotal->GetBinContent(i,j)/ei/ej);
     }
  }

  delete [] binMap;

  //========================
  // Step 3: plots

  TCanvas output;
  output.Divide(3,2);
  output.cd(1);
  histDetDATA->SetMinimum(0.0);
  histDetDATA->Draw("E");
  histDetMCall->SetMinimum(0.0);
  histDetMCall->SetLineColor(kBlue);  
  histDetMCbgr->SetLineColor(kRed);  
  histDetMC[1]->SetLineColor(kCyan);  
  histDetMCall->Draw("SAME HIST");
  histDetMCbgr->Draw("SAME HIST");
  histDetMC[1]->Draw("SAME HIST");

  output.cd(2);
  histUnfoldTotal->SetMinimum(0.0);
  histUnfoldTotal->SetMaximum(histUnfoldTotal->GetMaximum()*1.5);
  // outer error: total error
  histUnfoldTotal->Draw("E");
  // middle error: stat+bgr
  histUnfoldStatBgr->Draw("SAME E1");
  // inner error: stat only
  histUnfoldStat->Draw("SAME E1");
  histGenDATA->Draw("SAME HIST");
  histGenMC->SetLineColor(kBlue);
  histGenMC->Draw("SAME HIST");

  output.cd(3);
  histGenDet->SetLineColor(kBlue);
  histGenDet->Draw("BOX");

  // show tau as a function of chi**2
  output.cd(4);
  logTauX->Draw();
  bestLogTauLogChi2->SetMarkerColor(kRed);
  bestLogTauLogChi2->Draw("*");

  // show the L curve
  output.cd(5);
  lCurve->Draw("AL");
  bestLcurve->SetMarkerColor(kRed);
  bestLcurve->Draw("*");

  // show correlation matrix
  output.cd(6);
  histCorr->Draw("BOX");

  output.SaveAs("testUnfold3.ps");


  // step 4: compare results to
  // the so-called bin-by-bin "correction"
  for(Int_t i=1;i<=nGen;i++) {
     // data contribution in this bin
     Double_t data=histDetDATAbbb->GetBinContent(i);
     Double_t errData=histDetDATAbbb->GetBinError(i);

     // subtract background contribution
     Double_t data_bgr=data;
     Double_t errData_bgr=errData;
     for(Int_t j=1;j<=2;j++) {
        data_bgr -= scale_bgr*histDetMCbbb[j]->GetBinContent(i);
        errData_bgr = TMath::Sqrt(errData_bgr*errData_bgr+
                                  scale_bgr*histDetMCbbb[j]->GetBinError(i)*
                                  scale_bgr*histDetMCbbb[j]->GetBinError(i)+
                                  dscale_bgr*histDetMCbbb[j]->GetBinContent(i)*
                                  dscale_bgr*histDetMCbbb[j]->GetBinContent(i)
                                  );
     }
     // "correct" the data, using the Monte Carlo and neglecting off-diagonals
     Double_t fCorr=(histGenMC->GetBinContent(i)/
                      histDetMCbbb[0]->GetBinContent(i));
     Double_t data_bbb= data_bgr *fCorr;
     // stat only error
     Double_t errData_stat_bbb = errData*fCorr;
     // stat plus background subtraction
     Double_t errData_statbgr_bbb = errData_bgr*fCorr;
     // estimate systematic error by repeating the exercise
     // using the MC with systematic shifts applied
     Double_t fCorr_1=(histGenSYS1->GetBinContent(i)/
                       histDetSYS1bbb->GetBinContent(i));
     Double_t shift_sys1= data_bgr*fCorr_1 - data_bbb;
     Double_t fCorr_2=(histGenSYS2->GetBinContent(i)/
                        histDetSYS2bbb->GetBinContent(i));
     Double_t shift_sys2= data_bgr*fCorr_2 - data_bbb;

     cout<<data_bbb<<" "<<shift_sys1<<" "<<shift_sys2<<"\n";

     // add systematic shifts quadratically and get total error
     Double_t errData_total_bbb=
        TMath::Sqrt(errData_statbgr_bbb*errData_statbgr_bbb
                    +shift_sys1*shift_sys1
                    +shift_sys2*shift_sys2);

     // get results from real unfolding
     Double_t data_unfold= histUnfoldStat->GetBinContent(i);
     Double_t errData_stat_unfold=histUnfoldStat->GetBinError(i);
     Double_t errData_statbgr_unfold=histUnfoldStatBgr->GetBinError(i);
     Double_t errData_total_unfold=histUnfoldTotal->GetBinError(i);

     // compare
     std::cout<<"Bin "<<i<<": true "<<histGenDATA->GetBinContent(i)
              <<" unfold: "<<data_unfold
              <<" +/- "<<errData_stat_unfold<<" (stat)"
              <<" +/- "<<TMath::Sqrt(errData_statbgr_unfold*
                                     errData_statbgr_unfold-
                                     errData_stat_unfold*
                                     errData_stat_unfold)<<" (bgr)"
              <<" +/- "<<TMath::Sqrt(errData_total_unfold*
                                     errData_total_unfold-
                                     errData_statbgr_unfold*
                                     errData_statbgr_unfold)<<" (sys)"<<"\n";
     std::cout<<"Bin "<<i<<": true "<<histGenDATA->GetBinContent(i)
              <<" binbybin: "<<data_bbb
              <<" +/- "<<errData_stat_bbb<<" (stat)"
              <<" +/- "<<TMath::Sqrt(errData_statbgr_bbb*
                                     errData_statbgr_bbb-
                                     errData_stat_bbb*
                                     errData_stat_bbb)<<" (bgr)"
              <<" +/- "<<TMath::Sqrt(errData_total_bbb*
                                     errData_total_bbb-
                                     errData_statbgr_bbb*
                                     errData_statbgr_bbb)<<" (sys)"
              <<"\n";
  }
}
