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

#include "TGenerator.h"
#include "TAttParticle.h"
#include "TPrimary.h"
#include "TVenus.h"
#include "Vcommon.h"

#ifndef WIN32
#  define aaset aaset_
#  define atitle atitle_
#  define veanly veanly_
#  define ainit ainit_
#  define avenus avenus_
#  define qnbaaa qnbaaa_
#  define aseed aseed_
#  define afiles afiles_
#  define type_of_call
#else
#  define aaset AASET
#  define atitle ATITLE
#  define veanly VEANLY
#  define ainit  AINIT
#  define avenus AVENUS
#  define qnbaaa  QNBAAA
#  define aseed ASEED
#  define afiles AFILES
#  define type_of_call _stdcall
#endif

ClassImp(TVenus)


extern "C" {
//  Venus function calls
        void type_of_call aaset(Int_t &key);
        void type_of_call atitle();
        void type_of_call ainit();
        void type_of_call avenus();
        void type_of_call qnbaaa();
        void type_of_call aseed();
        void type_of_call afiles();
}

//______________________________________________________________________________
TVenus::TVenus():TGenerator("Venus","Venus")
{
//
//  Event generator Venus default constructor
//

}

//______________________________________________________________________________
TVenus::TVenus(char  *choice              , Int_t    /*numberOfEvents*/,
               Int_t    zprojectile       , Int_t    aprojectile,
               Int_t    ztarget           , Int_t    atarget,
               Double_t incidentMomentum,
               Bool_t   labSys            , Bool_t   lastGeneration,
               Double_t impactParameterMin, Double_t impactParameterMax,
               Double_t phiMin            , Double_t phiMax)
               :TGenerator("Venus","Venus")
{
//
//  Event generator Venus normal constructor
//

//
//  Zero initilisation and reset of all parameters
//

Int_t zero=0;
  aaset(zero);

  if (!strcmp(choice,"Analysis")) {
     PARO2.iappl = 0;
     fChoice = choice;
  }
  else if (!strcmp(choice,"Hadron")) {
     PARO2.iappl = 1;
     fChoice = choice;
  }
  else if (!strcmp(choice,"Geometry")) {
     PARO2.iappl = 2;
     fChoice = choice;
  }
  else if (!strcmp(choice,"Lepton")) {
     PARO2.iappl = 3;
     fChoice = choice;
  }
  else if (!strcmp(choice,"Cluster")) {
     PARO2.iappl = 4;
     fChoice = choice;
  }
  else {
     Error("Venus","Not a valid choice : %s, reset to Hadron",choice);
      PARO2.iappl = 1;
      fChoice = "Hadron";
  }

  if (aprojectile < 0) {
     Error("Venus","Negative A of projectile = %i, reset to 1 !",aprojectile);
     aprojectile = 1;
  }
  PARO2.maproj         = aprojectile;
  fAProjectile         = aprojectile;

  if (atarget < 0) {
     Error("Venus","Negative A of target = %i, reset to 1 !",atarget);
     atarget = 1;
  }
  PARO2.matarg         = atarget;
  fATarget             = atarget;

  if (zprojectile < 0) {
     Error("Venus","Negative Z of projectile = %i, reset to 1 !",zprojectile);
     zprojectile = 1;
  }
  PARO2.laproj         = zprojectile;
  fZProjectile         = zprojectile;

  if (ztarget < 0) {
     Error("Venus","Negative Z of target = %i, reset to 1 !",atarget);
     ztarget = 1;
  }
  PARO2.latarg         = ztarget;
  fZTarget             = ztarget;

  PARO1.labsys         = labSys;
  fLabSys              = labSys;
  PARO2.istmax         = !lastGeneration;
  fLastGeneration       = lastGeneration;

  if (incidentMomentum < 0.0) {
     Error("Venus","Negative incident hadron momentum = %d, reset to 200 !",
           incidentMomentum);
     incidentMomentum = 200.;
  }
  PARO2.pnll         = Float_t(incidentMomentum);
  fIncidentMomentum  = incidentMomentum;

  if (impactParameterMin < 0.0 || impactParameterMin > 100. ||
      impactParameterMin > impactParameterMax ) {
     Error("Venus","Negative impact parameter = %d, reset to 0 !",
           impactParameterMin);
     impactParameterMin = 0.0;
  }
  PAROI.bminim         = Float_t(impactParameterMin);
  fImpactParameterMin  = impactParameterMin;

  if (impactParameterMax < 0.0 || impactParameterMax > 10000. ||
      impactParameterMin > impactParameterMax ) {
     Error("Venus","Invalid impact parameter = %d, reset to 0 !",
           impactParameterMax);
     impactParameterMax = 0.0;
  }
  PAROI.bmaxim         = Float_t(impactParameterMax);
  fImpactParameterMax  = impactParameterMax;

  if (phiMin < 0.0 || phiMin > 2*TMath::Pi() || phiMin > phiMax) {
     Error("Venus","Invalid impact angles %d  %d, reset to 0 and 0 !",
           phiMin,phiMax);
     phiMin = 0.0;
     phiMax = 0.0;
  }
  fPhiMin      = phiMin;
  PAROI.phimin = phiMin;

  if (phiMax < 0.0 || phiMax > 2*TMath::Pi() || phiMin > phiMax) {
     Error("Venus","Invalid impact angles %d  %d, reset to 0 and 0 !",
           phiMin,phiMax);
     phiMin = 0.0;
     phiMax = 0.0;
  }
  fPhiMax      = phiMax;
  PAROI.phimax = phiMax;
//
//  Set some default values for the rest of the data members
//
  SetVersionNumber(521);
  SetU_D_QuarkProductionProb(PARO1.pud);
  SetQQ_QQbarProbability(PARO1.pdiqua);
  SetLightFlavoursSpinProb(PARO1.pspinl);
  SetHeavyFlavoursSpinProb(PARO1.pspinh);
  SetIsoSpinProb(PARO1.pispn);
  Setp_T_Distribution(PARO1.ioptf);
  SetAveragep_T(PARO1.ptf);
  SetStringTension(PARO1.tensn);
  SetStringDecayParameter(PARO1.parea);
  SetThresholdResonanceToString(PARO1.delrem);
  SetCutOffForKmaxor(PARO2.kutdiq);
  SetBreakingProcedureOption(PARO1.iopbrk);
  SetQuarkp_TDistributionOption(PARO1.ioptq);
  SetMeanTransverseQuarkMomentum(PARO8.ptq1,PARO8.ptq2,PARO8.ptq3);
  SetSemihardInteractionProb(PARO1.phard);
  SetSemihardCutOff(PARO1.pth);
  SetSeaRatio(PARO1.rstras);
  SetProjectileDiffractiveProb(PARO1.wproj);
  SetTargetDiffractiveProb(PARO1.wtarg);
  SetStructureFunctionSeaValence(PARO1.cutmsq);
  SetStructureFunctionCutOffMass(PARO1.cutmss);
  SetDiffractiveValenceQuarkFrac(PARO1.pvalen);
  SetPhaseSpace(PARO1.delmss);
  SetGribovReggeGamma(PARO4.grigam);
  SetGribovReggeRSquared(PARO4.grirsq);
  SetGribovReggeDelta(PARO4.gridel);
  SetGribovReggeSlope(PARO4.grislo);
  SetGribovReggeCrossSecWeight(PARO4.gricel);
  SetHardCoreDistance(PARO1.core);
  SetJ_PsiNucleonCrossSec(PARO1.sigj);
  SetReactionTime(PARO2.taurea);
  SetBaryonRadius(PARO9.radbar);
  SetMesonRadius(PARO9.radmes);
  SetInteractionMass(PARO1.amsiac);
  SetJIntaOption(PARO1.iojint);
  SetPrintOptionAmprif(PARO1.amprif);
  SetPrintOptionDelvol(PAROH.delvol);
  SetPrintOptionDeleps(PAROH.deleps);
  SetEntropyOption(PARO1.iopent,PARO3.uentro,PARO1.kentro);
  SetDecayTime(PARO1.taunll);
  SetOscillatorQuantum(PARO3.omega);
  SetSpaceTimeEvolutionMinTau(PARO1.taumin);
  SetTauSteps(PARO1.numtau);
  Setp_TDistributionRange(PARO1.ptmx);
  SetGaussDistributionRange(PARO1.gaumx);
  SetDensityDistributionRange(PARO1.fctrmx);
  SetTryAgain(PARO1.ntrymx);
  SetJ_PsiEvolutionTime(PARO1.taumx);
  SetJ_PsiEvolutionTimeSteps(PARO1.nsttau);
  SetMinimumEnergyOption(PARO1.iopenu);
  SetBergerJaffeTheta(PARO1.themas);
  SetSeaProbability(PARO2.prosea);
  SetInelasticProtonProtonCrossSec(PARO1.sigppi);
  SetEntropyCalculated(PARO2.ientro);
  SetDualPartonModel(PARO2.idpm);
  SetAntiQuarkColourExchange(PARO1.iaqu);
  SetMinNumberOfValenceQuarks(PARO1.neqmn);
  SetMaxNumberOfValenceQuarks(PARO1.neqmx);
  SetRapidityUpperLimit(PAROF.ymximi);
  SetClean(PARO1.nclean);
  SetCMToLabTransformation(PARO1.labsys);
  SetMaxNumberOfCollisions(PARO1.ncolmx);
  SetMaxResonanceSpin(PARO1.maxres);
  SetMomentumRescaling(PARO1.irescl);
  SetNueEnergy(PARO2.elepti);
  SetMuonEnergy(PARO2.elepto);
  SetMuonAngle(PARO2.angmue);
  SetCollisionTrigger(PARO1.ko1ko2);
  SetPrintOption(PARO2.ish);
  SetPrintSubOption(PARO2.ishsub);
  SetEventPrint(PARO2.ishevt);
  SetPrintMarks(PARO2.ipagi);
  SetMaxImpact(PAROI.bmaxim);
  SetMinImpact(PAROI.bminim);
  SetStoreOnlyStable(PARO2.istmax);
  SetInitialRandomSeed(CSEED.seedi);
  SetJFRADESuppression(PARO1.ifrade);
  SetResonanceStable(P13.ndecay);
  SetSpaceTimeEvolution(PAROG.ispall);
  SetMinTimeInEvolution(PAROG.wtmini);
  SetTimeStepInEvolution(PAROG.wtstep);
  SetCentralPointInEvolution(PAROG.iwcent);
  SetsMass(PARO8.smas);
  SetuuMass(PARO8.uumas);
  SetusMass(PARO8.usmas);
  SetssMass(PARO8.ssmas);
  SetStorage(PAROB.istore);

//
//  Finish initialization
//
  fCascadeStable = kFALSE;
  fKStable       = kFALSE;
  fLambdaStable  = kFALSE;
  fOmegaStable   = kFALSE;
  fPiZeroStable  = kFALSE;
  fSigmaStable   = kFALSE;
  atitle();
  afiles();
  fUpdate = kTRUE;

  SetTitle("Venus event");
}

//______________________________________________________________________________
TVenus::~TVenus()
{
//
//  Event generator VENUS default destructor
//

}

//______________________________________________________________________________
void TVenus::GenerateEvent(Option_t *option)
{
//
//  Generate next event
//
  if (fUpdate) {
     ainit();
     fUpdate = kFALSE;
  }
  if (fChoice.Contains("Analysis") || fChoice.Contains("Hadron") ||
      fChoice.Contains("Lepton")) {
    avenus();
  }
  else if (fChoice.Contains("Cluster")) {
    qnbaaa();
  }
  else {
     Error("GenerateEvent","Not a valid choice : %s, cannot generate event",
           fChoice.Data());
  }
  aseed();

  ImportParticles(option);
}

//______________________________________________________________________________
TObjArray* TVenus::ImportParticles(Option_t *option)
{
//
//  Overloaded primary creation method. The event generator does
//  not use the HEPEVT common block, and has not the PDG numbering scheme.
//
  fParticles->Delete();

    Int_t identifier;
  Int_t nptevt = 0;
  Int_t ipart  = 0;
  for (Int_t j = 0; j < CPTL.nptl; j++) {
     if (CPTL.istptl[j] <= PARO2.istmax) nptevt++;
  }
  Printf("Number of final particles : %d",nptevt);
  if (!strcmp(option,"") || !strcmp(option,"Final")) {
    for (Int_t i = 0; i<VCOMMON_MXPTL; i++) {
      if (CPTL.idptl[i] == 0) break;
      if (CPTL.istptl[i] <= PARO2.istmax) {
//
//  Use the common block values for the TPrimary constructor
//
        identifier = TAttParticle::ConvertISAtoPDG(CPTL.idptl[i]);
        if (identifier) {
           TPrimary *p = new TPrimary(
                                   identifier,
                                   -1,
                                   -1,
                                   CPTL.istptl[i],
                                   CPTL.pptl[i][0],
                                   CPTL.pptl[i][1],
                                   CPTL.pptl[i][2],
                                   CPTL.pptl[i][3],
                                   CPTL.xorptl[i][0],
                                   CPTL.xorptl[i][1],
                                   CPTL.xorptl[i][2],
                                   CPTL.xorptl[i][3],
                                   CPTL.tivptl[i][1]);
           fParticles->Add(p);
           ipart++;
        }
      }
    }
  }
  else if (!strcmp(option,"All")) {
    for (Int_t i = 0; i<VCOMMON_MXPTL; i++) {
      if (CPTL.idptl[i] != 0) {
        identifier = TAttParticle::ConvertISAtoPDG(CPTL.idptl[i]);
        if (identifier) {
           TPrimary *p = new TPrimary(
                                   identifier,
                                   CPTL.iorptl[i],
                                   CPTL.jorptl[i],
                                   CPTL.istptl[i],
                                   CPTL.pptl[i][0],
                                   CPTL.pptl[i][1],
                                   CPTL.pptl[i][2],
                                   CPTL.pptl[i][3],
                                   CPTL.xorptl[i][0],
                                   CPTL.xorptl[i][1],
                                   CPTL.xorptl[i][2],
                                   CPTL.xorptl[i][3],
                                   CPTL.tivptl[i][1]);
           fParticles->Add(p);
           ipart++;
        }
      }
    }
  }
  return fParticles;
}

//______________________________________________________________________________
Int_t TVenus::ImportParticles(TClonesArray *particles, Option_t *option)
{
//
//  Default primary creation method. It reads the /HEPEVT/ common block which
//  has been filled by the GenerateEvent method. If the event generator does
//  not use the HEPEVT common block, This routine has to be overloaded by
//  the subclasses.
//  The function loops on the generated particles and store them in
//  the TClonesArray pointed by the argument particles.
//  The default action is to store only the stable particles (ISTHEP = 1)
//  This can be demanded explicitly by setting the option = "Final"
//  If the option = "All", all the particles are stored.
//
   return TGenerator::ImportParticles(particles,option);
}

//______________________________________________________________________________
void TVenus::SetVersionNumber(Int_t iversn)
{
//
//  Set VENUS version number, the default is version 5.21
//
  if (iversn < 521) {
     Error("Venus","Version number too low = %d, reset to 521",
           iversn);
     iversn = 521;
  }
  fVersion           = iversn;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetU_D_QuarkProductionProb(Float_t pud)
{
//
//  Set VENUS ud quark production probability
//
  if (pud < 0.0) {
     Error("Venus",
           "VENUS ud quark production probability = %f, reset to 0.455",
           pud);
     pud = 0.455;
  }
  PARO1.pud        = pud;
  fPud             = pud;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetQQ_QQbarProbability(Float_t pdiqua)
{
//
//  Set VENUS qq - qqbar probability
//
  if (pdiqua < 0.0) {
     Error("Venus",
           "VENUS ud quark production probability = %f, reset to 0.455",
           pdiqua);
      pdiqua = 0.08;
  }
  PARO1.pdiqua        = pdiqua;
  fPDiQuark           = pdiqua;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetLightFlavoursSpinProb(Float_t pspinl)
{
//
//  Set VENUS light flavour spin probability
//
  if (pspinl < 0.0) {
     Error("Venus",
           "VENUS light flavour spin probability = %f, reset to 0.5",
          pspinl);
      pspinl = 0.5;
  }
  PARO1.pspinl        = pspinl;
  fPSpinLight         = pspinl;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetHeavyFlavoursSpinProb(Float_t pspinh)
{
//
//  Set VENUS heavy flavour spin probability
//
  if (pspinh < 0.0) {
     Error("Venus",
           "VENUS heavy flavour spin probability = %f, reset to 0.5",
          pspinh);
      pspinh = 0.5;
  }
  PARO1.pspinh        = pspinh;
  fPSpinHeavy         = pspinh;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetIsoSpinProb(Float_t pispn)
{
//
//  Set VENUS isospin probability
//
  if (pispn < 0.0) {
     Error("Venus",
           "VENUS isospin probability = %f, reset to 0.5",
          pispn);
     pispn = 0.5;
  }
  PARO1.pispn       = pispn;
  fPIsospin         = pispn;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::Setp_T_Distribution(Int_t ioptf)
{
//
//  Set VENUS transverse momentum distribution
//  ioptf = 1 -> exponential
//  ioptf = 2 -> gaussian
//
  if (ioptf < 1 || ioptf > 2) {
     Error("Venus",
           "VENUS invalid pt distribution parameter = %d, reset to 1 ",
          ioptf);
     ioptf = 1;
  }
  PARO1.ioptf       = ioptf;
  fPTDistribution   = ioptf;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetAveragep_T(Float_t ptf)
{
//
//  Set VENUS average transverse momentum
//
  if (ptf < 0.0) {
     Error("Venus",
           "VENUS invalid average transverse momentum = %f, reset to 0.45 ",
          ptf);
     ptf = 0.5;
  }
  PARO1.ptf       = ptf;
  fAveragePt      = ptf;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetStringTension(Float_t tensn)
{
//
//  Set VENUS string tension
//
  if (tensn < 0.0) {
     Error("Venus",
           "VENUS invalid string tension = %f, reset to 1.0 ",
          tensn);
     tensn = 0.5;
  }
  PARO1.tensn       = tensn;
  fStringTension    = tensn;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetStringDecayParameter(Float_t parea)
{
//
//  Set VENUS string decay parameter
//
  if (parea < 0.0) {
     Error("Venus",
           "VENUS invalid string decay parameter = %f, reset to 0.20 ",
          parea);
     parea = 0.20;
  }
  PARO1.parea       = parea;
  fStringDecay      = parea;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetThresholdResonanceToString(Float_t delrem)
{
//
//  Set VENUS threshold for resonance to string
//
  if (delrem < 0.0) {
     Error("Venus",
       "VENUS invalid threshold for resonance to string = %f, reset to 1.0 ",
       delrem);
     delrem = 1.0;
  }
  PARO1.delrem       = delrem;
  fResThreshold      = delrem;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetCutOffForKmaxor(Int_t kutdiq)
{
//
//  Set VENUS cut off for Kmaror
//
  if (kutdiq < 0) {
     Error("Venus",
           "VENUS invalid cutoff value for Kmaxor = %d, reset to 4 ",
           kutdiq);
     kutdiq = 4;
  }
  PARO2.kutdiq    = kutdiq;
  fCutKmaxor      = kutdiq;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetBreakingProcedureOption(Int_t iopbrk)
{
//
//  Set VENUS breaking procedure option
//  iopbrk = 1 -> amor
//  iopbrk = 2 -> samba
//
  if (iopbrk < 1 || iopbrk > 2) {
     Error("Venus",
           "VENUS invalid breaking procedure option = %d, reset to 1 ",
           iopbrk);
     iopbrk = 1;
  }
  PARO1.iopbrk    = iopbrk;
  fBreaking       = iopbrk;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetQuarkp_TDistributionOption(Int_t ioptq)
{
//
//  Set VENUS quark p_T distribution option
//  ioptq = 1 -> exponential
//  ioptq = 2 -> gaussian
//  ioptq = 3 -> power
//
  if (ioptq < 1 || ioptq > 3) {
     Error("Venus",
           "VENUS invalid Quark transverse momentum option = %d, reset to 2 ",
           ioptq);
     ioptq = 2;
  }
  PARO1.ioptq    = ioptq;
  fQuarkPt       = ioptq;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMeanTransverseQuarkMomentum(Float_t ptq1,
                                            Float_t ptq2,
                                            Float_t ptq3)
{
//
//  Set VENUS mean quark transverse momentum
//  with the energy dependence of mean transverse momentum of quarks:
//  ptq1+ptq2*alog(e)+ptq3*alog(e)**2
//  with e=sqrt(s)
//
  if (ptq1 < 0.0) {
     Error("Venus",
"VENUS invalid mean transverse quark momentum component 1 = %f, reset to 0.26 ",
ptq1);
     ptq1 = .26;
  }
  PARO8.ptq1            = ptq1;
  fMeanQMomentum1       = ptq1;
  if (ptq2 < 0.0) {
     Error("Venus",
"VENUS invalid mean transverse quark momentum component 2 = %f, reset to 0.0 ",
ptq2);
     ptq2 = 0.0;
  }
  PARO8.ptq2            = ptq2;
  fMeanQMomentum2       = ptq2;
  if (ptq3 < 0.0) {
     Error("Venus",
"VENUS invalid mean transverse quark momentum component 3 = %f, reset to 0.0 ",
ptq3);
     ptq1 = 0.0;
  }
  PARO8.ptq3            = ptq3;
  fMeanQMomentum3       = ptq3;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetSemihardInteractionProb(Float_t phard)
{
//
//  Set VENUS semihard interaction probability (not used if negative)
//
  PARO1.phard    = phard;
  fSemihard      = phard;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetSemihardCutOff(Float_t pth)
{
//
//  Set VENUS cutoff parameter for p_t distr for semihard interactions
//
  if (pth < 0.0) {
      Error("Venus",
           "Invalid parameter for semihard interactions = %f, reset to 1.0",
           pth);
      pth = 0.;
  }
  PARO1.pth     = pth;
  fCutSemi      = pth;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetSeaRatio(Float_t rstras)
{
//
//  Set VENUS ratio of strang sea quark contents over up sea contents
//
  if (rstras < 0.0) {
      Error("Venus",
            "Invalid sea ratio = %f, reset to 0.0",
            rstras);
      rstras = 0.;
  }
  PARO1.rstras     = rstras;
  fSeaRatio        = rstras;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetProjectileDiffractiveProb(Float_t wproj)
{
//
//  Set VENUS projectile diffractive probability
//
  if (wproj < 0.0) {
      Error("Venus",
            "Invalid projectile diffractive probability = %f, reset to 0.32",
            wproj);
      wproj = 0.32;
  }
  PARO1.wproj     = wproj;
  fProjDiffProb   = wproj;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetTargetDiffractiveProb(Float_t wtarg)
{
//
//  Set VENUS target diffractive probability
//
  if (wtarg < 0.0) {
  Error("Venus","Invalid target diffractive probability = %f, reset to 0.32",
            wtarg);
      wtarg = 0.32;
  }
  PARO1.wtarg     = wtarg;
  fTargDiffProb   = wtarg;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetStructureFunctionSeaValence(Float_t cutmsq)
{
//
//  Set VENUS effective cutoff mass in structure fcts for sea-valence ratio
//
  if (cutmsq < 0.0) {
      Error("Venus",
      "Invalid cutoff mass in structure fcts = %f, reset to 2.0",
      cutmsq);
      cutmsq = 2.0;
  }
  PARO1.cutmsq     = cutmsq;
  fSeaValenceCut   = cutmsq;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetStructureFunctionCutOffMass(Float_t cutmss)
{
//
//  Set VENUS effective cutoff mass in structure fcts for sea-valence ratio
//
  if (cutmss < 0.0) {
            Error("Venus",
            "Invalid cutoff mass  for sea-valence ratio= %f, reset to 0.001",
            cutmss);
      cutmss = 2.0;
  }
  PARO1.cutmss     = cutmss;
  fCutMass         = cutmss;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetDiffractiveValenceQuarkFrac(Float_t pvalen)
{
//
//  Set VENUS valence quark fraction in case of diffractive interaction
//
  if (pvalen < 0.0) {
  Error("Venus","Invalid valence quark fraction = %f, reset to 0.30",
            pvalen);
      pvalen = .30;
  }
  PARO1.pvalen         = pvalen;
  fValenceFrac         = pvalen;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetPhaseSpace(Float_t delmss)
{
//
//  Set VENUS phase space parameter
//
  if (delmss < 0.0) {
      Error("Venus",
            "Invalid phase space parameter = %f, reset to 0.30",
            delmss);
      delmss = .30;
  }
  PARO1.delmss        = delmss;
  fPhaseSpace         = delmss;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetGribovReggeGamma(Float_t grigam)
{
//
//  Set VENUS gribov-regge-theory gamma
//
  if (grigam < 0.0) {
      Error("Venus",
            "Invalid gribov-regge-theory gamma = %f, reset to 3.64*0.04",
            grigam);
      grigam = 3.64*0.04;
  }
  PARO4.grigam         = grigam;
  fGribovGamma         = grigam;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetGribovReggeRSquared(Float_t grirsq)
{
//
//  Set VENUS gribov-regge-theory r^2
//
  if (grirsq < 0.0) {
      Error("Venus",
            "Invalid gribov-regge-theory r squared = %f, reset to 3.56*0.04",
            grirsq);
      grirsq = 3.56*0.04;
  }
  PARO4.grirsq      = grirsq;
  fGribovR2         = grirsq;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetGribovReggeDelta(Float_t gridel)
{
//
//  Set VENUS gribov-regge-theory delta
//
  if (gridel < 0.0) {
      Error("Venus",
            "Invalid gribov-regge-theory delta = %f, reset to 3.56*0.04",
            gridel);
      gridel = 0.07;
  }
  PARO4.gridel         = gridel;
  fGribovDelta         = gridel;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetGribovReggeSlope(Float_t grislo)
{
//
//  Set VENUS gribov-regge-theory slope
//
  if (grislo < 0.0) {
      Error("Venus",
         "Invalid gribov-regge-theory slope = %f, reset to 0.25*0.04",
         grislo);
      grislo = 0.25*0.04;
  }
  PARO4.grislo         = grislo;
  fGribovSlope         = grislo;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetGribovReggeCrossSecWeight(Float_t gricel)
{
//
//  Set VENUS gribov-regge-theory cross section weight
//
  if (gricel < 0.0) {
      Error("Venus",
         "Invalid gribov-regge-theory cross section weight = %f, reset to 1.5",
         gricel);
      gricel = 1.5;
  }
  PARO4.gricel         = gricel;
  fGribovCross         = gricel;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetHardCoreDistance(Float_t core)
{
//
//  Set VENUS hard core distance
//
  if (core < 0.0) {
      Error("Venus",
            "Invalid hard core distance = %f, reset to 0.8",
            core);
      core = 0.8;
  }
  PARO1.core         = core;
  fCore              = core;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetJ_PsiNucleonCrossSec(Float_t sigj)
{
//
//  Set VENUS J/Psi nucleon cross section
//
  if (sigj < 0.0) {
      Error("Venus",
            "Invalid J/Psi nucleon cross section = %f, reset to 0.2",
            sigj);
      sigj = 0.2;
  }
  PARO1.sigj               = sigj;
  fSigmaJ_Psi              = sigj;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetReactionTime(Float_t taurea)
{
//
//  Set VENUS reaction time
//
  if (taurea < 0.0) {
      Error("Venus",
            "Invalid reaction time = %f, reset to 1.5",
            taurea);
      taurea = 1.5;
  }
  PARO2.taurea           = taurea;
  fReacTime              = taurea;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetBaryonRadius(Float_t radbar)
{
//
//  Set VENUS baryon radius
//
  if (radbar < 0.0) {
      Error("Venus",
            "Invalid baryon radius = %f, reset to 0.63",
            radbar);
      radbar = 0.63;
  }
  PARO9.radbar           = radbar;
  fBaryonRadius          = radbar;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMesonRadius(Float_t radmes)
{
//
//  Set VENUS meson radius
//
  if (radmes < 0.0) {
      Error("Venus",
            "Invalid meson radius = %f, reset to 0.40",
            radmes);
      radmes = 0.40;
  }
  PARO9.radmes          = radmes;
  fMesonRadius          = radmes;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetInteractionMass(Float_t amsiac)
{
//
//  Set VENUS interaction mass
//
  if (amsiac < 0.0) {
      Error("Venus",
            "Invalid interaction mass = %f, reset to 0.8",
            amsiac);
      amsiac = 0.8;
  }
  PARO1.amsiac          = amsiac;
  fInteractionMass      = amsiac;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetJIntaOption(Int_t iojint)
{
//
//  Set VENUS option to call jinta1 (1) or jinta2 (2)
//
  if (iojint < 1 || iojint > 2) {
      Error("Venus",
            "Invalid option to call jintaN = %d, reset to 2",
            iojint);
      iojint = 2;
  }
  PARO1.iojint          = iojint;
  fJintalOpt            = iojint;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetPrintOptionAmprif(Float_t amprif)
{
//
//  Set VENUS print option
//
  if (amprif < 0.0) {
      Error("Venus",
            "Invalid print option = %g, reset to 0",
            amprif);
      amprif = 0.0;
  }
  PARO1.amprif             = amprif;
  fPrintOption1            = (Int_t)amprif;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetPrintOptionDelvol(Float_t delvol)
{
//
//  Set VENUS print option
//
  if (delvol < 0.0) {
      Error("Venus",
            "Invalid print option = %g, reset to 1",
            delvol);
      delvol = 1.0;
  }
  PAROH.delvol             = delvol;
  fPrintOption2            = (Int_t)delvol;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetPrintOptionDeleps(Float_t deleps)
{
//
//  Set VENUS print option
//
  if (deleps < 0.0) {
      Error("Venus",
            "Invalid print option = %g, reset to 1",
            deleps);
      deleps = 1.0;
  }
  PAROH.deleps             = deleps;
  fPrintOption3            = (Int_t)deleps;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetEntropyOption(Int_t   iopent,
                              Float_t uentro,
                              Int_t kentro)
{
//
//  Set VENUS options for entropy calculation
//  option for entropy calculation
//  iopent = 0 -> zero entropy
//  iopent = 1 -> oscillator model (0 for k.le.uentro)
//  iopent = 2 -> fermi gas w const volume (0 for k.le.uentro)
//  iopent = 3 -> fermi gas w const density (0 for k.le.uentro)
//  iopent = 4 -> fermi gas w const vol - new (0 for k.le.uentro)
//  iopent = 5 -> resonance gas (hagedorn) (0 for u.le.uentro)
//
  if (iopent < 0) {
      Error("Venus",
            "Invalid option for entropy calculation = %d, reset to 5",
            iopent);
      iopent = 5;
  }
  PARO1.iopent             = iopent;
  fEntropy1                = iopent;
  if (uentro < 0.0) {
      Error("Venus",
            "Invalid option for entropy calculation = %d, reset to 4.0",
            uentro);
      uentro = 5;
  }
  PARO3.uentro             = uentro;
  fEntropy2                = uentro;
  if (kentro < 0) {
      Error("Venus",
            "Invalid option for entropy calculation = %d, reset to 10000",
            kentro);
      kentro = 10000;
  }
  PARO1.kentro             = kentro;
  fEntropy3                = kentro;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetDecayTime(Float_t taunll)
{
//
//  Set VENUS decay time of comoving frame
//
  if (taunll < 0) {
      Error("Venus",
            "Invalid decay time = %f, reset to 1.0",
            taunll);
      taunll = 1.0;
  }
  PARO1.taunll          = taunll;
  fDecayTime            = taunll;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetOscillatorQuantum(Float_t omega)
{
//
//  Set VENUS oscillator quantum
//
  if (omega < 0.) {
      Error("Venus",
            "Invalid oscillator quantum = %f, reset to 0.5",
            omega);
      omega = 0.5;
  }
  PARO3.omega          = omega;
  fOsciQuantum         = omega;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetSpaceTimeEvolutionMinTau(Float_t taumin)
{
//
//  Set VENUS minimum tau for space-time evolution
//
  if (taumin < 0.) {
      Error("Venus",
            "Invalid minimum tau for space-time evolution = %f, reset to 1.0",
            taumin);
      taumin = 0.5;
  }
  PARO1.taumin    = taumin;
  fTauMin         = taumin;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetTauSteps(Int_t numtau)
{
//
//  Set VENUS number of tau steps
//
  if (numtau < 0) {
      Error("Venus",
            "Invalid number of tau steps = %d, reset to 86",
            numtau);
      numtau = 86;
  }
  PARO1.numtau      = numtau;
  fTauSteps         = numtau;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::Setp_TDistributionRange(Float_t ptmx)
{
//
//  Set VENUS transverse momentum distribution range
//
  if (ptmx < 0.) {
      Error("Venus",
            "Invalid transverse momentum distribution range = %f, reset to 6.0",
            ptmx);
      ptmx = 6.0;
  }
  PARO1.ptmx       = ptmx;
  fPtRange         = ptmx;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetGaussDistributionRange(Float_t gaumx)
{
//
//  Set VENUS gaussian distribution range
//
  if (gaumx < 0.) {
      Error("Venus",
            "Invalid gaussian distribution range = %f, reset to 8.0",
            gaumx);
      gaumx = 8.0;
  }
  PARO1.gaumx       = gaumx;
  fGaussianRange    = gaumx;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetDensityDistributionRange(Float_t fctrmx)
{
//
//  Set VENUS density distribution range
//
  if (fctrmx < 0.) {
      Error("Venus",
            "Invalid density distribution range = %f, reset to 10.0",
            fctrmx);
      fctrmx = 10.0;
  }
  PARO1.fctrmx     = fctrmx;
  fDensityRange    = fctrmx;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetTryAgain(Int_t ntrymx)
{
//
//  Set VENUS number of retries
//
  if (ntrymx < 0) {
      Error("Venus",
            "Invalid density distribution range = %d, reset to 10",
            ntrymx);
      ntrymx = 10;
  }
  PARO1.ntrymx     = ntrymx;
  fRetries         = ntrymx;
  fUpdate = kTRUE;
}


//______________________________________________________________________________
void TVenus::SetJ_PsiEvolutionTime(Float_t taumx)
{
//
//  Set VENUS J/Psi evolution time
//
  if (taumx < 0) {
      Error("Venus",
            "Invalid J/Psi evolution time = %f, reset to 20.0",
            taumx);
      taumx = 20.0;
  }
  PARO1.taumx             = taumx;
  fJ_PsiEvolution         = taumx;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetJ_PsiEvolutionTimeSteps(Int_t nsttau)
{
//
//  Set VENUS J/Psi evolution time steps
//
  if (nsttau < 0) {
      Error("Venus",
            "Invalid number of J/Psi evolution time steps = %d, reset to 100",
            nsttau);
      nsttau = 100;
  }
  PARO1.nsttau             = nsttau;
  fJ_PsiSteps              = nsttau;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMinimumEnergyOption(Int_t iopenu)
{
//
//  Set VENUS minimum energy option
//  iopent = 1 -> sum of hadron masses
//  iopent = 2 -> bag model curve with minimum at nonzero strangen
//
  if (iopenu < 1 || iopenu > 2) {
      Error("Venus",
            "Invalid minimum energy option = %d, reset to 1",
            iopenu);
      iopenu = 1;
  }
  PARO1.iopenu             = iopenu;
  fMinEnergy               = iopenu;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetBergerJaffeTheta(Float_t themas)
{
//
//  Set VENUS parameter theta in berger/jaffe mass formula
//
  if (themas < 0.0) {
      Error("Venus",
            "Invalid minimum energy option = %f, reset to 0.51225",
            themas);
      themas = 0.51225;
  }
  PARO1.themas             = themas;
  fMassTheta               = themas;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetSeaProbability(Float_t prosea)
{
//
//  Set VENUS sea prob (if < 0 then calculated from structure fncts)
//
  PARO2.prosea             = prosea;
  fSeaProbability          = prosea;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetInelasticProtonProtonCrossSec(Float_t sigppi)
{
//
//  Set VENUS inelastic pp cross section [fm**2]
//  if negative: calculated from gribov-regge-theory
//
  PARO1.sigppi             = sigppi;
  fPPInelastic             = sigppi;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetEntropyCalculated(Int_t ientro)
{
//
//  Set VENUS entro() calculated (1) or from data (2)
//
  if (ientro < 1 || ientro > 2) {
      Error("Venus",
            "Invalid entropy calculation option = %d, reset to 2",
            ientro);
      ientro = 2;
  }
  PARO2.ientro             = ientro;
  fEntropyCalc             = ientro;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetDualPartonModel(Int_t idpm)
{
//
//  Set VENUS dual parton model (1) or not (else)
//
  if (idpm < 0) {
      Error("Venus",
            "Invalid dual parton model selection option = %d, reset to 0",
            idpm);
      idpm = 0;
  }
  PARO2.idpm             = idpm;
  fDualParton            = idpm;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetAntiQuarkColourExchange(Int_t iaqu)
{
//
//  Set VENUS antiquark color exchange (1) or not (0)
//
  if (iaqu < 0 || iaqu > 1) {
      Error("Venus",
            "Invalid antiquark color exchange option = %d, reset to 1",
            iaqu);
      iaqu = 1;
  }
  PARO1.iaqu             = iaqu;
  fColourExchange        = iaqu;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMinNumberOfValenceQuarks(Int_t neqmn)
{
//
//  Set VENUS minimum number of valence quarks
//
  if (neqmn > 0) {
      Error("Venus",
            "Invalid minimum number of valence quarks = %d, reset to -5",
            neqmn);
      neqmn = -5;
  }
  PARO1.neqmn       = neqmn;
  fMinValence       = neqmn;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMaxNumberOfValenceQuarks(Int_t neqmx)
{
//
//  Set VENUS maximum number of valence quarks
//
  if (neqmx < 0) {
      Error("Venus",
            "Invalid maximum number of valence quarks = %d, reset to 5",
            neqmx);
      neqmx = 5;
  }
  PARO1.neqmx       = neqmx;
  fMaxValence       = neqmx;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetRapidityUpperLimit(Float_t ymximi)
{
//
//  Set VENUS upper limit for rapidity interval for intermittency analysis
//
  if (ymximi < 0.0) {
      Error("Venus",
            "Invalid upper limit for rapidity interval = %f, reset to 2.0",
            ymximi);
      ymximi = 2.0;
  }
  PAROF.ymximi       = ymximi;
  fInterval          = ymximi;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetClean(Int_t nclean)
{
//
//  Set VENUS clean /cptl/ if nclean > 0 (every nclean_th time step)
//
  if (nclean < 0) {
      Error("Venus",
            "Invalid clean step number = %d, reset to 0",
            nclean);
      nclean = 0;
  }
  PARO1.nclean       = nclean;
  fClean             = nclean;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetCMToLabTransformation(Int_t labsys)
{
//
//  Set VENUS trafo from pp-cm into lab-system (1) or not (else)
//
  if (labsys < 0) {
      Error("Venus",
            "Invalid CMS to Laboratory system transformation = %d, reset to 1",
            labsys);
      labsys = 1;
  }
  PARO1.labsys        = labsys;
  fLabSys             = labsys;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMaxNumberOfCollisions(Int_t ncolmx)
{
//
//  Set VENUS maximum number of collisions
//
  if (ncolmx < 0) {
      Error("Venus",
            "Invalid maximum number of collisions = %d, reset to 10000",
            ncolmx);
      ncolmx = 10000;
  }
  PARO1.ncolmx        = ncolmx;
  fMaxCollisions      = ncolmx;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMaxResonanceSpin(Int_t maxres)
{
//
//  Set VENUS maximum resonance spin
//
  if (maxres < 0) {
      Error("Venus",
            "Invalid maximum resonance spin = %d, reset to 99999",
            maxres);
      maxres = 99999;
  }
  PARO1.maxres        = maxres;
  fMaxSpin            = maxres;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMomentumRescaling(Int_t irescl)
{
//
//  Set VENUS momentum rescaling (1) or not (else)
//
  if (irescl < 0) {
      Error("Venus",
            "Invalid momentum rescaling parameter = %d, reset to 1",
            irescl);
      irescl = 1;
  }
  PARO1.irescl        = irescl;
  fRescale            = irescl;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetNueEnergy(Float_t elepti)
{
//
//  Set VENUS nue energy
//
  if (elepti < 0.0) {
      Error("Venus",
            "Invalid nue energy = %f, reset to 43.00",
            elepti);
      elepti = 43.00;
  }
  PARO2.elepti        = elepti;
  fNueEnergy          = elepti;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMuonEnergy(Float_t elepto)
{
//
//  Set VENUS muon energy
//
  if (elepto < 0.0) {
      Error("Venus",
            "Invalid muon energy = %f, reset to 26.24",
            elepto);
      elepto = 26.24;
  }
  PARO2.elepto         = elepto;
  fMuonEnergy          = elepto;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMuonAngle(Float_t angmue)
{
//
//  Set VENUS muon angle
//
  if (angmue < 0.0) {
      Error("Venus",
            "Invalid muon angle = %f, reset to 3.9645/360.0*2*3.14159",
            angmue);
      angmue = 3.9645/360.0*2*3.14159;
  }
  PARO2.angmue         = angmue;
  fMuonAngle           = angmue;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetCollisionTrigger(Int_t ko1ko2)
{
//
//  Set VENUS collision trigger (only coll between ko1 and ko2 are used)
//
  if (ko1ko2 < 0) {
      Error("Venus",
            "Invalid collision trigger = %d, reset to 9999",
            ko1ko2);
      ko1ko2 = 9999;
  }
  PARO1.ko1ko2         = ko1ko2;
  fCollisionTrigger    = ko1ko2;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetPrintOption(Int_t ish)
{
//
//  Set VENUS print option
//    ish=0      option for printout
//        14 -> call uttima
//        16 -> prints sea prob.
//        18 -> sr jclude, no-phase-space droplets
//        19 -> sr ainitl, call smassp
//        21 -> creates histogram for sea distribution
//        22 -> sr jfrade, msg after call utclea
//        23 -> call jintfp
//        24 -> call jintcl
//        25 -> call jchprt
//        26 -> call qgcpen (plot of energy and flavor density)
//        27 -> call qnbcor
//        29 -> call qgcpfl (plot of dispersion)
//        90,91,92,93,94,95 -> more and more detailed messages.
//
  if (ish != 0  && ish != 14 && ish != 16 && ish != 18 && ish != 19 &&
      ish != 21 && ish != 22 && ish != 23 && ish != 24 && ish != 25 &&
      ish != 26 && ish != 27 && ish != 29 && ish != 90 && ish != 91 &&
      ish != 92 && ish != 93 && ish != 94 && ish != 95) {
      Error("Venus",
            "Invalid print option = %d, reset to 0",
            ish);
      ish = 0;
  }
  PARO2.ish         = ish;
  fPrintOption      = ish;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetPrintSubOption(Int_t ishsub)
{
//
//  Set VENUS print sub option
//      ishsub=0   option for printout
//      ishsub=ijmn, ij specifies location where ish=mn.
//             ij=01 -> sr jcludc
//             ij=02 -> sr jetgen
//             ij=03 -> sr jfrade, starting before fragmentation
//             ij=04 -> sr jdecay
//             ij=05 -> sr jdecax
//             ij=06 -> sr nucoll
//             ij=07 -> sr nucoge+-
//             ij=08 -> sr aastor
//             ij=09 -> sr jfrade, starting after fragmentation
//             ij=10 -> sr jfrade, starting before decay
//             ij=11 -> sr jfrade, starting after interactions
//             ij=12 -> sr jcentr, entro() in data format
//             ij=13 -> sr jcentp
//             ij=14 -> sr jdecax if droplet decay
//             ij=15 -> sr jsplit
//             ij=16 -> sr jfrade
//             ij=17 -> sr racpro
//             ij=18 -> sr utclea
//             ij=19 -> sr jinta1, jinta2, after call utclea
//             ij=20 -> sr jdecas
//             ij=21 -> sr jdecas (without jdecax)
//             ij=22 -> sr utcley
//             ij=50 -> sr qgcnbi
//
  if ((ishsub < 0 || ishsub > 22) && ishsub != 50) {
      Error("Venus",
            "Invalid print sub option = %d, reset to 0",
            ishsub);
      ishsub = 0;
  }
  PARO2.ishsub         = ishsub;
  fPrintSubOption      = ishsub;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetEventPrint(Int_t ishevt)
{
//
//  Set VENUS ishevt != 0: for evt# != ishevt ish is set to 0
//
  if (ishevt < 0) {
      Error("Venus",
            "Invalid request to print event = %d, reset to 0",
            ishevt);
      ishevt = 0;
  }
  PARO2.ishevt   = ishevt;
  fPrintEvent    = ishevt;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetPrintMarks(Int_t ipagi)
{
//
//  Set VENUS print marks between whom ish is set to ish(init)
//
  if (ipagi < 0) {
      Error("Venus",
            "Invalid request to print marks = %d, reset to 0",
            ipagi);
      ipagi = 0;
  }
  PARO2.ipagi    = ipagi;
  fPrintMarks    = ipagi;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMaxImpact(Float_t bmaxim)
{
//
//  Set VENUS maximum beam impact parameter
//
  if (bmaxim < 0.0 || bmaxim > 10000.) {
     Error("Venus","Invalid maximum impact parameter = %f, reset to 10000.",
           bmaxim);
     bmaxim = 10000.0;
  }
  PAROI.bmaxim         = bmaxim;
  fImpactParameterMax  = bmaxim;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMinImpact(Float_t bminim)
{
//
//  Set VENUS minimum beam impact parameter
//
  if (bminim < 0.0 || bminim > 10000.) {
     Error("Venus","Invalid minimum impact parameter = %f, reset to 0.0!",
           bminim);
     bminim = 0.0;
  }
  PAROI.bminim         = bminim;
  fImpactParameterMin  = bminim;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetStoreOnlyStable(Int_t istmax)
{
//
//  Set VENUS store only stable ptl (0) or also parents (1)
//
  if (istmax < 0 || istmax > 1) {
     Error("Venus",
           "Invalid request for stable partcile storage = %d, reset to 0",
           istmax);
     istmax = 0;
  }
  PARO2.istmax         = istmax;
  fLastGeneration      = !istmax;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetInitialRandomSeed(Double_t seedi)
{
//
//  Set VENUS initial random number seed
//
  CSEED.seedi         = seedi;
  fInitialSeed        = seedi;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetJFRADESuppression(Int_t ifrade)
{
//
//  Set VENUS suppression of calling jfrade (0). jfrade=fragm+decay+rescatt
//
  if (ifrade < 0 || ifrade > 1) {
     Error("Venus",
           "Invalid suppression in calling jfrade = %d, reset to 0",
           ifrade);
     ifrade = 1;
  }
  PARO1.ifrade         = ifrade;
  fJfradeSup           = ifrade;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetResonanceStable(Bool_t stable)
{
//
//  Set VENUS all resonance decays suppressed
//
  if (stable) {
     P13.ndecay = 1;
  }
  else {
     P13.ndecay = 0;
  }
  fAllStable = stable;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetKShortKLongStable(Bool_t stable)
{
//
//  Set VENUS k_short/long (+-20) decays suppressed
//
  if (stable && !fKStable) {
     P13.ndecay += 10;
  }
  else if (!stable && fKStable) {
     P13.ndecay -= 10;
  }
  fKStable = stable;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetLambdaStable(Bool_t stable)
{
//
//  Set VENUS lambda (+-2130) decays suppressed
//
  if (stable && !fLambdaStable) {
     P13.ndecay += 100;
  }
  else if (!stable && fLambdaStable) {
     P13.ndecay -= 100;
  }
  fLambdaStable = stable;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetSigmaStable(Bool_t stable)
{
//
//  Set VENUS sigma (+-1130,+-2230) decays suppressed
//
  if (stable && !fSigmaStable) {
     P13.ndecay += 1000;
  }
  else if (!stable && fSigmaStable) {
     P13.ndecay -= 1000;
  }
  fSigmaStable = stable;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetCascadeStable(Bool_t stable)
{
//
//  Set VENUS cascade (+-2330,+-1330) decays suppressed
//
  if (stable && !fCascadeStable) {
     P13.ndecay += 10000;
  }
  else if (!stable && fCascadeStable) {
     P13.ndecay -= 10000;
  }
  fCascadeStable = stable;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetOmegaStable(Bool_t stable)
{
//
//  Set VENUS omega (+-3331) decays suppressed
//
  if (stable && !fOmegaStable) {
     P13.ndecay += 100000;
  }
  else if (!stable && fOmegaStable) {
     P13.ndecay -= 100000;
  }
  fOmegaStable = stable;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetPiZeroStable(Bool_t stable)
{
//
//  Set VENUS pi0 (110) decays suppressed
//
  if (stable && !fPiZeroStable) {
     P13.ndecay += 1000000;
  }
  else if (!stable && fPiZeroStable) {
     P13.ndecay -= 1000000;
  }
  fPiZeroStable = stable;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetRhoPhiRatio(Float_t rhophi)
{
//
//  Set VENUS rho/rho+phi ratio
//
  if (rhophi < 0.0) {
     Error("Venus",
           "Invalid rho/rho+phi ratio = %f, reset to 0.5",
           rhophi);
     rhophi = 0.5;
  }
  PARO2.rhophi         = rhophi;
  fRhoPhi              = rhophi;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetSpaceTimeEvolution(Int_t ispall)
{
//
//  Set VENUS wspa: all ptls (1) or only interacting ptls (else)
//
  if (ispall < 0) {
     Error("Venus",
           "Invalid space time evolution paramter = %d, reset to 1",
           ispall);
     ispall = 1;
  }
  PAROG.ispall         = ispall;
  fSpaceTime           = ispall;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetMinTimeInEvolution(Float_t wtmini)
{
//
//  Set VENUS tmin in wspa
//
  PAROG.wtmini         = wtmini;
  fMinTime             = wtmini;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetTimeStepInEvolution(Float_t wtstep)
{
//
//  Set VENUS t-step in wspa
//
  if (wtstep < 0) {
     Error("Venus",
           "Invalid space time steps = %f, reset to 1.0",
           wtstep);
     wtstep = 1.0;
  }
  PAROG.wtstep         = wtstep;
  fSpaceTimeStep       = wtstep;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetCentralPointInEvolution(Int_t iwcent)
{
//
//  Set VENUS only central point (1) or longitudinal distr (else) in wspa
//
  if (iwcent < 0) {
     Error("Venus",
           "Invalid central point parameter = %d, reset to 0",
           iwcent);
     iwcent = 0;
  }
  PAROG.iwcent         = iwcent;
  fCentralPoint        = iwcent;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetsMass(Float_t smas)
{
//
//  Set VENUS s quark mass
//
  if (smas < 0) {
     Error("Venus",
           "Invalid s quark mass = %f, reset to 0.0",
           smas);
     smas = 0.0;
  }
  PARO8.smas         = smas;
  fSQuarkMass        = smas;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetuuMass(Float_t uumas)
{
//
//  Set VENUS uu diquark mass
//
  if (uumas < 0) {
     Error("Venus",
           "Invalid uu diquark quark mass = %f, reset to 0.0",
           uumas);
     uumas = 0.0;
  }
  PARO8.uumas         = uumas;
  fuuQuarkMass        = uumas;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetusMass(Float_t usmas)
{
//
//  Set VENUS us diquark mass
//
  if (usmas < 0) {
     Error("Venus",
           "Invalid us diquark quark mass = %f, reset to 0.0",
           usmas);
     usmas = 0.0;
  }
  PARO8.usmas         = usmas;
  fusQuarkMass        = usmas;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetssMass(Float_t ssmas)
{
//
//  Set VENUS ss diquark mass
//
  if (ssmas < 0) {
     Error("Venus",
           "Invalid ss diquark quark mass = %f, reset to 0.0",
           ssmas);
     ssmas = 0.0;
  }
  PARO8.ssmas         = ssmas;
  fssQuarkMass        = ssmas;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::SetStorage(Int_t istore)
{
//
//  Set VENUS internal storage parameter
//
  if (istore < 0 || istore > 2) {
     Error("Venus","Invalid internal storage parameter = %d, reset to 0 !",
           istore);
     istore = 0;
  }
  PAROB.istore       = istore;
  fIstore            = istore;
  fUpdate = kTRUE;
}

//______________________________________________________________________________
void TVenus::Show()
{
//
//  Shows the actual values of some common block variables as in
//  the original VENUS command 'show'
//
   Printf("iversn %d",CVSN.iversn);
   Printf("iappl  %d",PARO2.iappl);
   Printf("nevent %d",PARO2.nevent);
   Printf("iprmpt %d",PAROB.iprmpt);
   Printf("ish    %d",PARO2.ish);
   Printf("ishsub %d",PARO2.ishsub);
   Printf("irandm %d",PARO2.irandm);
   Printf("irewch %d",PARO2.irewch);
   Printf("ishevt %d",PARO2.ishevt);
   Printf("iecho  %d",PAROB.iecho);
   Printf("modsho %d",PARO2.modsho);
   Printf("idensi %d",PARO7.idensi);
   Printf("pud    %f",PARO1.pud);
   Printf("pdiqua %f",PARO1.pdiqua);
   Printf("pspinl %f",PARO1.pspinl);
   Printf("pspinh %f",PARO1.pspinh);
   Printf("pispn  %f",PARO1.pispn);
   Printf("ioptf  %d",PARO1.ioptf);
   Printf("ptf    %f",PARO1.ptf);
   Printf("ptmx   %f",PARO1.ptmx);
   Printf("tensn  %f",PARO1.tensn);
   Printf("parea  %f",PARO1.parea);
   Printf("delrem %f",PARO1.delrem);
   Printf("delrex %f",PARO1.delrex);
   Printf("kutdiq %i",PARO2.kutdiq);
   Printf("iopbrk %d",PARO1.iopbrk);
   Printf("smas   %f",PARO8.smas);
   Printf("uumas  %f",PARO8.uumas);
   Printf("usmas  %f",PARO8.usmas);
   Printf("ssmas  %f",PARO8.ssmas);
   Printf("ndecay %d",P13.ndecay);
   Printf("maxres %d",PARO1.maxres);
   Printf("rhophi %f",PARO2.rhophi);
   Printf("engy   %f",PARO2.engy);
   Printf("elepti %f",PARO2.elepti);
   Printf("elepto %f",PARO2.elepto);
   Printf("angmue %f",PARO2.angmue);
   Printf("pnll   %f",PARO2.pnll);
   Printf("idproj %d",PARO2.idproj);
   Printf("idtarg %d",PARO2.idtarg);
   Printf("ioptq  %d",PARO1.ioptq);
   Printf("ptq1   %f",PARO8.ptq1);
   Printf("ptq2   %f",PARO8.ptq2);
   Printf("ptq3   %f",PARO8.ptq3);
   Printf("phard  %f",PARO1.phard);
   Printf("pth    %f",PARO1.pth);
   Printf("rstras %f",PARO1.rstras);
   Printf("wproj  %f",PARO1.wproj);
   Printf("wtarg  %f",PARO1.wtarg);
   Printf("cutmsq %f",PARO1.cutmsq);
   Printf("cutmss %f",PARO1.cutmss);
   Printf("pvalen %f",PARO1.pvalen);
   Printf("delmss %f",PARO1.delmss);
   Printf("neqmn  %d",PARO1.neqmn);
   Printf("neqmx  %d",PARO1.neqmx);
   Printf("iaqu   %d",PARO1.iaqu);
   Printf("prosea %f",PARO2.prosea);
   Printf("idpm   %d",PARO2.idpm);
   Printf("iopadi %d",PAROA.iopadi);
   Printf("q2soft %f",PAROA.q2soft);
   Printf("grigam %f",PARO4.grigam);
   Printf("grirsq %f",PARO4.grirsq);
   Printf("gridel %f",PARO4.gridel);
   Printf("grislo %f",PARO4.grislo);
   Printf("gricel %f",PARO4.gricel);
   Printf("sigppi %f",PARO1.sigppi);
   Printf("laproj %d",PARO2.laproj);
   Printf("maproj %d",PARO2.maproj);
   Printf("latarg %d",PARO2.latarg );
   Printf("matarg %d",PARO2.matarg);
   Printf("core   %f",PARO1.core);
   Printf("ncolmx %d",PARO1.ncolmx);
   Printf("fctrmx %f",PARO1.fctrmx);
   Printf("ko1ko2 %d",PARO1.ko1ko2);
   Printf("bmaxim %f",PAROI.bmaxim);
   Printf("bminim %f",PAROI.bminim);
   Printf("phimax %f",PAROI.phimax);
   Printf("phimin %f",PAROI.phimin);
   Printf("taurea %f",PARO2.taurea);
   Printf("sigmes %f",PARO9.sigmes);
   Printf("sigbar %f",PARO9.sigbar);
   Printf("rinmes %f",PARO9.rinmes);
   Printf("rinbar %f",PARO9.rinbar);
   Printf("epscri %f",PARO9.epscri);
   Printf("amsiac %f",PARO1.amsiac);
   Printf("iojint %d",PARO1.iojint);
   Printf("amprif %f",PARO1.amprif);
   Printf("delvol %f",PAROH.delvol);
   Printf("deleps %f",PAROH.deleps);
   Printf("taumin %f",PARO1.taumin);
   Printf("deltau %f",PARO1.deltau);
   Printf("factau %f",PARO1.factau);
   Printf("numtau %d",PARO1.numtau);
   Printf("etafac %f",PAROH.etafac);
   Printf("dlzeta %f",PAROH.dlzeta);
   Printf("ioclud %d",PARO7.ioclud);
   Printf("corlen %f",PARO5.corlen);
   Printf("dezzer %f",PARO5.dezzer);
   Printf("amuseg %f",PARO5.amuseg);
   Printf("bag4rt %f",PARO5.bag4rt);
   Printf("iopent %d",PARO1.iopent);
   Printf("uentro %f",PARO3.uentro);
   Printf("kentro %d",PARO1.kentro);
   Printf("taunll %f",PARO1.taunll);
   Printf("omega  %f",PARO3.omega);
   Printf("ientro %d",PARO2.ientro);
   Printf("tecm   %f",CONFIG.tecm);
   Printf("volu   %f",CONFIG.volu);
   Printf("keu    %d",CINFLA.keu);
   Printf("ked    %d",CINFLA.ked);
   Printf("kes    %d",CINFLA.kes);
   Printf("kec    %d",CINFLA.kec);
   Printf("keb    %d",CINFLA.keb);
   Printf("ket    %d",CINFLA.ket);
   Printf("iterma %d",CITER.iterma);
   Printf("iterpr %d",CITER.iterpr);
   Printf("iterpl %d",CITER.iterpl);
   Printf("iternc %d",CITER.iternc );
   Printf("iospec %d",PARO6.iospec );
   Printf("iocova %d",PARO6.iocova);
   Printf("iopair %d",PARO6.iopair);
   Printf("iozero %d",PARO6.iozero );
   Printf("ioflac %d",PARO6.ioflac);
   Printf("iomom  %d",PARO6.iomom);
   Printf("iograc %d",PARO7.iograc);
   Printf("iocite %d",PARO7.iocite);
   Printf("ioceau %d",PARO7.ioceau);
   Printf("iociau %d",PARO7.iociau);
   Printf("nadd   %d",PARO7.nadd);
   Printf("epsr   %f",CEPSR.epsr);
   Printf("keepr  %d",CMETRO.keepr);
   Printf("iopenu %d",PARO1.iopenu);
   Printf("themas %f",PARO1.themas);
   Printf("jpsi   %d",PARO2.jpsi);
   Printf("jpsifi %d",PARO2.jpsifi);
   Printf("sigj   %f",PARO1.sigj);
   Printf("taumx  %f",PARO1.taumx);
   Printf("nsttau %d",PARO1.nsttau);
   Printf("ijphis %d",PARO2.ijphis);
   Printf("ymximi %f",PAROF.ymximi);
   Printf("imihis %d",PAROF.imihis);
   Printf("isphis %d",PAROG.isphis);
   Printf("ispall %d",PAROG.ispall);
   Printf("wtmini %f",PAROG.wtmini);
   Printf("wtstep %f",PAROG.wtstep);
   Printf("iwcent %d",PAROG.iwcent);
   Printf("iclhis %d",PAROF.iclhis);
   Printf("iwtime %d",PAROF.iwtime);
   Printf("wtimet %f",PAROF.wtimet);
   Printf("wtimei %f",PAROF.wtimei);
   Printf("wtimea %f",PAROF.wtimea);
   Printf("gaumx  %f",PARO1.gaumx);
   Printf("nclean %d",PARO1.nclean);
   Printf("istore %d",PAROB.istore);
   Printf("labsys %d",PARO1.labsys);
   Printf("irescl %d",PARO1.irescl);
   Printf("ifrade %d",PARO1.ifrade);
   Printf("ntrymx %d",PARO1.ntrymx);
   Printf("istmax %d",PARO2.istmax);
   Printf("seedi  %f",CSEED.seedi);
}
