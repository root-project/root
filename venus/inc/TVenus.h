// @(#)root/venus:$Name$:$Id$
// Author: Ola Nordmann   21/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TVenus                                                                   //
//                                                                          //
// TVenus is an interface class between the event generator VENUS and       //
// the ROOT system. The current implementation is based on VENUS 5.21       //
//                                                                          //
// Authors of VENUS:                                                        //
//           Klaus WERNER                                                   //
//           Eugen FURLER (upgrade 5.04)                                    //
//           Michael HLADIK (upgrades 5.05, 5.13, 5.15, 5.17)               //
//                                                                          //
//  Laboratoire Subatech, Universite de Nantes - IN2P3/CNRS - EMN           //
//  4 rue Alfred Kastler, 44070 Nantes Cedex 03, France                     //
//                                                                          //
//  Email: <last name>@nanhp3.in2p3.fr                                      //
//                                                                          //
//  VENUS is a Monte Carlo procedure to simulate hadronic interactions at   //
//  ultrarelativistic energies (hadron-hadron, hadron-nucleus, nucleus-     //
//  nucleus scattering).                                                    //
//                                                                          //
//  VENUS is based on Gribov-Regge theory (of multiple Pomeron exchange)    //
//  and classical relativistic string dynamics. A detailed description can  //  //
//  be found in a review article, published in Physics Reports 232 (1993)   //
//  pp. 87-299.                                                             //
//                                                                          //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVenus
#define ROOT_TVenus

#ifndef ROOT_TGenerator
#include "TGenerator.h"
#endif

class TVenus : public TGenerator {

protected:
  Bool_t         fAllStable;
  Int_t          fAProjectile;
  Double_t       fAveragePt;
  Int_t          fATarget;
  Double_t       fBaryonRadius;
  Int_t          fBreaking;
  Bool_t         fCascadeStable;
  Int_t          fCentralPoint;
  TString        fChoice;
  Int_t          fClean;
  Int_t          fCollisionTrigger;
  Int_t          fColourExchange;
  Double_t       fCore;
  Int_t          fCutKmaxor;
  Double_t       fCutMass;
  Double_t       fCutSemi;
  Double_t       fDecayTime;
  Double_t       fDensityRange;
  Int_t          fDualParton;
  Int_t          fEntropy1;
  Double_t       fEntropy2;
  Int_t          fEntropy3;
  Int_t          fEntropyCalc;
  Double_t       fGaussianRange;
  Double_t       fGribovCross;
  Double_t       fGribovDelta;
  Double_t       fGribovGamma;
  Double_t       fGribovR2;
  Double_t       fGribovSlope;
  Double_t       fImpactParameterMin;
  Double_t       fImpactParameterMax;
  Double_t       fInitialSeed;
  Double_t       fInteractionMass;
  Double_t       fInterval;
  Int_t          fIstore;
  Int_t          fJfradeSup;
  Int_t          fJintalOpt;
  Double_t       fJ_PsiEvolution;
  Int_t          fJ_PsiSteps;
  Bool_t         fKStable;
  Bool_t         fLabSys;
  Bool_t         fLambdaStable;
  Bool_t         fLastGeneration;
  Double_t       fMassTheta;
  Int_t          fMaxCollisions;
  Int_t          fMaxSpin;
  Int_t          fMaxValence;
  Double_t       fMeanQMomentum1;
  Double_t       fMeanQMomentum2;
  Double_t       fMeanQMomentum3;
  Double_t       fMesonRadius;
  Int_t          fMinEnergy;
  Double_t       fMinTime;
  Int_t          fMinValence;
  Double_t       fMuonAngle;
  Double_t       fMuonEnergy;
  Double_t       fNueEnergy;
  Int_t          fNumberOfEvents;
  Double_t       fIncidentMomentum;
  Bool_t         fOmegaStable;
  Double_t       fOsciQuantum;
  Double_t       fPIsospin;
  Bool_t         fPiZeroStable;
  Double_t       fPhaseSpace;
  Double_t       fPhiMin;
  Double_t       fPhiMax;
  Double_t       fPDiQuark;
  Double_t       fPPInelastic;
  Int_t          fPrintEvent;
  Int_t          fPrintMarks;
  Int_t          fPrintOption;
  Int_t          fPrintOption1;
  Int_t          fPrintOption2;
  Int_t          fPrintOption3;
  Int_t          fPrintSubOption;
  Double_t       fProjDiffProb;
  Double_t       fPSpinLight;
  Double_t       fPSpinHeavy;
  Int_t          fPTDistribution;
  Double_t       fPtRange;
  Double_t       fPud;
  Int_t          fQuarkPt;
  Double_t       fReacTime;
  Int_t          fRescale;
  Double_t       fResThreshold;
  Int_t          fRetries;
  Double_t       fRhoPhi;
  Double_t       fSeaProbability;
  Double_t       fSeaRatio;
  Double_t       fSeaValenceCut;
  Double_t       fSemihard;
  Double_t       fSigmaJ_Psi;
  Bool_t         fSigmaStable;
  Int_t          fSpaceTime;
  Double_t       fSpaceTimeStep;
  Double_t       fssQuarkMass;
  Double_t       fSQuarkMass;
  Double_t       fStringDecay;
  Double_t       fStringTension;
  Double_t       fTargDiffProb;
  Double_t       fTauMin;
  Double_t       fTauMax;
  Int_t          fTauSteps;
  Bool_t         fUpdate;
  Double_t       fuuQuarkMass;
  Double_t       fusQuarkMass;
  Double_t       fValenceFrac;
  Int_t          fVersion;
  Int_t          fZProjectile;
  Int_t          fZTarget;

public:
  TVenus();
  TVenus(char *choice, Int_t numberOfEvents, Int_t zprojectile,
         Int_t aprojectile, Int_t ztarget, Int_t atarget,
         Double_t incidentMomentum, Bool_t labSys,
         Bool_t   lastGeneration, Double_t impactParameterMin,
         Double_t impactParameterMax, Double_t phiMin, Double_t phiMax);
  virtual ~TVenus();
  virtual void            GenerateEvent(Option_t *option="");
  virtual Int_t           ImportParticles(TClonesArray *particles, Option_t *option="");
  virtual TObjArray      *ImportParticles(Option_t *option="");
  virtual void            SetVersionNumber(Int_t iversn = 521);
  virtual void            SetU_D_QuarkProductionProb(Float_t pud = 0.455);
  virtual void            SetQQ_QQbarProbability(Float_t pdiqua = 0.08);
  virtual void            SetLightFlavoursSpinProb(Float_t pspinl = 0.50);
  virtual void            SetHeavyFlavoursSpinProb(Float_t pspinh = 0.75);
  virtual void            SetIsoSpinProb(Float_t pispn = 0.50);
  virtual void            Setp_T_Distribution(Int_t ioptf = 1);
  virtual void            SetAveragep_T(Float_t ptf = 0.45);
  virtual void            SetStringTension(Float_t tensn = 1.0);
  virtual void            SetStringDecayParameter(Float_t parea = 0.20);
  virtual void            SetThresholdResonanceToString(Float_t delrem = 1.0);
  virtual void            SetCutOffForKmaxor(Int_t kutdiq = 4);
  virtual void            SetBreakingProcedureOption(Int_t iopbrk = 1);
  virtual void            SetQuarkp_TDistributionOption(Int_t ioptq = 2);
  virtual void            SetMeanTransverseQuarkMomentum(Float_t ptq1 = 0.260,
                                                         Float_t ptq2 = 0.0,
                                                         Float_t ptq3 = 0.0);
  virtual void            SetSemihardInteractionProb(Float_t phard = -1.0);
  virtual void            SetSemihardCutOff(Float_t pth = 1.0);
  virtual void            SetSeaRatio(Float_t rstras = 0.0);
  virtual void            SetProjectileDiffractiveProb(Float_t wproj = 0.32);
  virtual void            SetTargetDiffractiveProb(Float_t wtarg = 0.32);
  virtual void            SetStructureFunctionSeaValence(Float_t cutmsq = 2.0);
  virtual void            SetStructureFunctionCutOffMass(Float_t cutmss = 0.001);
  virtual void            SetDiffractiveValenceQuarkFrac(Float_t pvalen = 0.30);
  virtual void            SetPhaseSpace(Float_t delmss = 0.300);
  virtual void            SetGribovReggeGamma(Float_t grigam = 3.64*0.04);
  virtual void            SetGribovReggeRSquared(Float_t grirsq = 3.56*0.04);
  virtual void            SetGribovReggeDelta(Float_t gridel = 0.07);
  virtual void            SetGribovReggeSlope(Float_t grislo = 0.25*0.04);
  virtual void            SetGribovReggeCrossSecWeight(Float_t gricel = 1.5);
  virtual void            SetHardCoreDistance(Float_t core = 0.8);
  virtual void            SetJ_PsiNucleonCrossSec(Float_t sigj = 0.2);
  virtual void            SetReactionTime(Float_t taurea = 1.5);
  virtual void            SetBaryonRadius(Float_t radbar = 0.63);
  virtual void            SetMesonRadius(Float_t radmes = 0.40);
  virtual void            SetInteractionMass(Float_t amsiac = 0.8);
  virtual void            SetJIntaOption(Int_t iojint = 2);
  virtual void            SetPrintOptionAmprif(Float_t amprif = 0.0);
  virtual void            SetPrintOptionDelvol(Float_t delvol = 1.0);
  virtual void            SetPrintOptionDeleps(Float_t deleps = 1.0);
  virtual void            SetEntropyOption(Int_t   iopent = 5,
                                           Float_t uentro = 4.,
                                           Int_t kentro = 10000);
  virtual void            SetDecayTime(Float_t taunll = 1.0);
  virtual void            SetOscillatorQuantum(Float_t omega = .500);
  virtual void            SetSpaceTimeEvolutionMinTau(Float_t taumin = 1.);
  virtual void            SetTauSteps(Int_t numtau = 86);
  virtual void            Setp_TDistributionRange(Float_t ptmx = 6.0);
  virtual void            SetGaussDistributionRange(Float_t gaumx = 8.0);
  virtual void            SetDensityDistributionRange(Float_t fctrmx = 10.0);
  virtual void            SetTryAgain(Int_t ntrymx = 10);
  virtual void            SetJ_PsiEvolutionTime(Float_t taumx = 20.0);
  virtual void            SetJ_PsiEvolutionTimeSteps(Int_t nsttau = 100);
  virtual void            SetMinimumEnergyOption(Int_t iopenu = 1);
  virtual void            SetBergerJaffeTheta(Float_t themas = 0.51225);
  virtual void            SetSeaProbability(Float_t prosea = -1.0);
  virtual void            SetInelasticProtonProtonCrossSec(Float_t sigppi = -1.0);
  virtual void            SetEntropyCalculated(Int_t ientro = 2);
  virtual void            SetDualPartonModel(Int_t idpm = 0);
  virtual void            SetAntiQuarkColourExchange(Int_t iaqu = 1);
  virtual void            SetMinNumberOfValenceQuarks(Int_t neqmn = -5);
  virtual void            SetMaxNumberOfValenceQuarks(Int_t neqmx = 5);
  virtual void            SetRapidityUpperLimit(Float_t ymximi = 2.0);
  virtual void            SetClean(Int_t nclean = 0);
  virtual void            SetCMToLabTransformation(Int_t labsys = 1);
  virtual void            SetMaxNumberOfCollisions(Int_t ncolmx = 10000);
  virtual void            SetMaxResonanceSpin(Int_t maxres = 99999);
  virtual void            SetMomentumRescaling(Int_t irescl = 1);
  virtual void            SetNueEnergy(Float_t elepti = 43.00);
  virtual void            SetMuonEnergy(Float_t elepto = 26.24);
  virtual void            SetMuonAngle(Float_t angmue = 3.9645/360.0*2*3.14159);
  virtual void            SetCollisionTrigger(Int_t ko1ko2 = 9999);
  virtual void            SetPrintOption(Int_t ish = 0);
  virtual void            SetPrintSubOption(Int_t ishsub = 0);
  virtual void            SetEventPrint(Int_t ishevt = 0);
  virtual void            SetPrintMarks(Int_t ipagi = 0);
  virtual void            SetMaxImpact(Float_t bmaxim = 10000.);
  virtual void            SetMinImpact(Float_t bminim = 0.);
  virtual void            SetStoreOnlyStable(Int_t istmax = 0);
  virtual void            SetInitialRandomSeed(Double_t seedi = 0.0);
  virtual void            SetJFRADESuppression(Int_t ifrade = 1);
  virtual void            SetResonanceStable(Bool_t stable = kFALSE);
  virtual void            SetKShortKLongStable(Bool_t stable);
  virtual void            SetLambdaStable(Bool_t stable);
  virtual void            SetSigmaStable(Bool_t stable);
  virtual void            SetCascadeStable(Bool_t stable);
  virtual void            SetOmegaStable(Bool_t stable);
  virtual void            SetPiZeroStable(Bool_t stable);
  virtual void            SetRhoPhiRatio(Float_t rhophi = 0.5);
  virtual void            SetSpaceTimeEvolution(Int_t ispall = 1);
  virtual void            SetMinTimeInEvolution(Float_t wtmini = -3.0);
  virtual void            SetTimeStepInEvolution(Float_t wtstep = 1.0);
  virtual void            SetCentralPointInEvolution(Int_t iwcent = 0);
  virtual void            SetsMass(Float_t smas = 0.0);
  virtual void            SetuuMass(Float_t uumas = 0.0);
  virtual void            SetusMass(Float_t usmas = 0.0);
  virtual void            SetssMass(Float_t ssmas = 0.0);
  virtual void            SetStorage(Int_t istore);
  virtual void            Show();

  ClassDef(TVenus,1)  //Venus event generator interface class
};

#endif
