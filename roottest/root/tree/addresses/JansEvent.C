// JansEvent.cc
//
// Implementation of the ROOT tuple

//WLAV: changed "BetaMiniUser/JansEvent.hh" -> "JansEvent.h"
#include "JansEvent.h"
#include "TMath.h"
#include "TFile.h"

ClassImp(CandidateParameters);
ClassImp(CandParametersMC);
ClassImp(B_GammaParameters);
ClassImp(B_ProtonParameters);
ClassImp(LambdaPionParameters);
ClassImp(LambdaProtonParameters);
ClassImp(LambdaParameters);
ClassImp(B_Parameters);
ClassImp(JansEventHeader);
ClassImp(JansEvent);

CandidateParameters::CandidateParameters()
{
    reset();
}

void CandidateParameters::reset()
{
    uid = 0;
    momentum.SetXYZ(0., 0., 0.);
}


CandParametersMC::CandParametersMC()
{
    reset();
}


void CandParametersMC::reset()
{
    pdgID = 0;
    parentPdgID = 0;
}

B_GammaParameters::B_GammaParameters()
{
    reset();
}

void B_GammaParameters::reset()
{
    CandidateParameters::reset();
//     CandParametersMC::reset();
    minNeutralDistance = 1000;
    minTrackDistance = 1000;
    minTrackDTheta = 1000;
    cmsEnergy = 0;
    labEnergy = 0;
    bestEtaMass = 100;
    bestPi0Mass = 100;
}


B_ProtonParameters::B_ProtonParameters()
{
    reset();
}

void B_ProtonParameters::reset()
{
    CandidateParameters::reset();
//     CandParametersMC::reset();
    charge = 0;
//    isPID = 0;
    pidMask = 0;
}


LambdaPionParameters::LambdaPionParameters()
{
    reset();
}

void LambdaPionParameters::reset()
{
    CandidateParameters::reset();
//     CandParametersMC::reset();
    charge = 0;
}



LambdaProtonParameters::LambdaProtonParameters()
{
    reset();
}

void LambdaProtonParameters::reset()
{
    CandidateParameters::reset();
//     CandParametersMC::reset();
    charge = 0;
//    isPID = 0;
    pidMask = 0;
}


LambdaParameters::LambdaParameters()
{
    reset();
}


void LambdaParameters::reset()
{
    CandidateParameters::reset();
    proton.reset();
    pion.reset();
    doca2IP = 0;
    mass = 0;
    decayLength = 0;
    decayLengthError = 0;
    chi2 = 0;
    nDOF = 0;
    decayVector.SetXYZ(0., 0., 0.);
}

float LambdaParameters::getProb()
{
    return TMath::Prob(chi2, nDOF);
}

B_Parameters::B_Parameters()
{
    reset();
}


void B_Parameters::reset()
{
    CandidateParameters::reset();
    lambda.reset();
    proton.reset();
    gamma.reset();
    isFromB_Status = 0;
    btaB_Status = 0;
    mMiss = 0;
    mES = 0;
    deltaE = 0;
    mass = 0;
    chi2 = 0;
    nDOF = 0;
    charge = 0;
    legendreRatio = 0;
    cosAngleB2Rest = 0;
    bThrustMag = 0;
    restOfEventThrustMag = 0;
}

float B_Parameters::getProb()
{
    return TMath::Prob(chi2, nDOF);
}

float B_Parameters::getAbsCosAngleB2Rest()
{
    return fabs(cosAngleB2Rest);
}


JansEventHeader::JansEventHeader()
{
    reset();
}


void JansEventHeader::reset()
{
    for (int i=0; i<NumFinalStates; ++i) {
        kidsFoundDaughter[i] = 0;
        kidsFoundList[i] = 0;
        kidsFoundRaw[i] = 0;
    }
    isWrongDecay = 0;
    isFSR_Decay = 0;
    // b1decayString.SetString("");
    // b2decayString.SetString("");
    hasBGFMultiHadronTag = 0;
    hasLambdaVeryVeryLooseTag = 0;
    r2 = 0;
    eventCMS.SetXYZT(0., 0., 0., 0.);
}

float JansEventHeader::getSqrtS()
{
    return eventCMS.E();
}

JansEvent::JansEvent() : bList("B_Parameters")
{
    reset();
}

void JansEvent::reset()
{
    eventHeader.reset();
    bList.Clear();
}
#include "Riostream.h"
#include "TTree.h"

void testJan() {
  {
    B_Parameters *b = new B_Parameters;
    //TObject *o = b;
    //CandidateParameters *c = b;
    //std::cout << (void*)b << " : " << (void*)c << " and " << (void*)o << endl;
    std::cout << b->gamma.minTrackDTheta << std::endl;
    std::cout << b->gamma.uid << std::endl;
    std::cout << b->gamma.GetName() << std::endl;
  }
  TFile* f = new TFile("janbug.root");
  TTree* t; f->GetObject("evtTree2",t);
  JansEvent* j = new JansEvent();
  t->SetBranchAddress("event", &j);
  t->GetEvent(0);
  std::cout << j->bList.GetEntries() << std::endl;
  B_Parameters *b = dynamic_cast<B_Parameters*>(j->bList[0]);
  //std::cout << (void*)j->bList[0] << " vs " << (void*)b << endl;
  std::cout << b->gamma.minTrackDTheta << std::endl;
  std::cout << b->gamma.uid << std::endl;
  std::cout << b->gamma.GetName() << std::endl;
}



#if defined(__MAKECINT__)
#pragma link C++ class CandidateParameters;
#pragma link C++ class CandParametersMC;
#pragma link C++ class LambdaParameters;
#pragma link C++ class B_Parameters;
#pragma link C++ class B_ProtonParameters;
#pragma link C++ class B_GammaParameters;
#pragma link C++ class LambdaPionParameters;
#pragma link C++ class LambdaProtonParameters;
#pragma link C++ class JansEvent;
#pragma link C++ class JansEventHeader;
#endif
