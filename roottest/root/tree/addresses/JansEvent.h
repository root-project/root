//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class MyMiniAnalysis - the barest outline of a Beta
//      Analysis, suitable for simple filling-in
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Bob Jacobsen                    Original author
//
// Copyright Information:
//      Copyright (C) 1997              Lawrence Berkeley Laboratory
//
//------------------------------------------------------------------------

#ifndef JANSEVENT_HH
#define JANSEVENT_HH


//----------------------
// Base Class Headers --
//----------------------
#include <TObject.h>
#include <TObjString.h>
#include <TVector3.h>
#include <TLorentzVector.h>
#include <TClonesArray.h>
#include <map>
#include <string>
#include <vector>

//WLAV: commented out the following line:
//#include "GClonesArray.hh"

// better handling of the single particles with truth matching
enum EFinalState {LambdaProton, LambdaPion, B_Proton, B_Gamma, NumFinalStates};

// basic BtaCandidate info
struct CandidateParameters
{
    CandidateParameters();
    //virtual 
        void reset();
    unsigned int uid; 
    // mind the implications of using vector for tracks
    TVector3 momentum;
};


// additional MC info
struct CandParametersMC
{
    CandParametersMC();
    void reset();
    long int pdgID;
    long int parentPdgID;
};


struct B_GammaParameters : public CandidateParameters, public TObject
{
    B_GammaParameters();
    void reset();
    // distance to the nearest Neutral in the Calorimeter
    float minNeutralDistance;
    // distance to the nearest Track in the Calorimeter
    float minTrackDistance;
    // distance to nearest track in theta
    float minTrackDTheta;
    // cms energy
    float cmsEnergy;
    float labEnergy;
    // the closest we can get to an eta with a combination of any other photon
    float bestEtaMass;
    // the closest we can get to a pi0 with a combination of any other photon
    float bestPi0Mass;
    ClassDefOverride(B_GammaParameters, 1);
};


struct B_ProtonParameters : public CandidateParameters, public TObject
{
    B_ProtonParameters();
    void reset();
    char charge; //!
//    unsigned char isPID;
    int pidMask;
    ClassDefOverride(B_ProtonParameters, 1);
};


struct LambdaPionParameters : public CandidateParameters, public TObject
{
    LambdaPionParameters();
    void reset();
    char charge; //!
    ClassDefOverride(LambdaPionParameters, 1);
};


struct LambdaProtonParameters : public CandidateParameters, public TObject
{
    LambdaProtonParameters();
    void reset();
    char charge; //!
//    unsigned char isPID;
    int pidMask;
    ClassDefOverride(LambdaProtonParameters, 1);
};


// because a Lambda is a composite, I don't need pdgID
struct LambdaParameters : public CandidateParameters, public TObject
{
    LambdaParameters();
    void reset();
    float getProb();
    LambdaProtonParameters proton;
    LambdaPionParameters pion;
    
    float doca2IP;
    float mass;
    float chi2; // vertex chi2
    int nDOF; // number degrees of freedom
    float decayLength;
    float decayLengthError; // significance of the length
    TVector3 decayVector;
    ClassDefOverride(LambdaParameters, 1);
};


// because a B is a composite, I don't need pdgID
struct B_Parameters : public CandidateParameters, public TObject
{
    B_Parameters();
    void reset();
    float getProb();
    float getAbsCosAngleB2Rest();
    // This one is in the Y(4S) rest frame
    B_GammaParameters gamma;
    B_ProtonParameters proton;
    LambdaParameters lambda;

    // to save bitset status - which of the daughters are truth matched
    unsigned short isFromB_Status;
    int btaB_Status;
    float mES;
    float mMiss;
    float deltaE;
    float mass;
    float chi2; // vertex chi2
        int nDOF;
    char charge;
    // L12/L10
    float legendreRatio;
    // those three are in the Upsilon4S rest frame
    float cosAngleB2Rest;
    float bThrustMag;
    float restOfEventThrustMag;
    ClassDefOverride(B_Parameters, 2);
};


struct JansEventHeader : public TObject
{
    JansEventHeader();
    void reset();
    float getSqrtS();

    // Match MC with B Daughter List
    unsigned char kidsFoundDaughter[NumFinalStates]; //! don't write to file

    // Match MC with reco list
    unsigned char kidsFoundList[NumFinalStates]; //!

    // Match MC with ChargedTracks/CalorNeutral
    unsigned char kidsFoundRaw[NumFinalStates]; //!

    unsigned char isWrongDecay;

    // Need vairiable to account for fsr signal (B- -> lambda pbar gamma gamma)
    unsigned char isFSR_Decay;

    // Truth info of decay
    // TObjString b1decayString;
    // TObjString b2decayString;

    // Tags
    unsigned char hasBGFMultiHadronTag;
    unsigned char hasLambdaVeryVeryLooseTag;
    float r2;
    TLorentzVector eventCMS;
    ClassDefOverride(JansEventHeader, 1);
};


struct JansEvent : public TObject
{
//WLAV: commented out the following three lines:
//#if !defined(__CINT__)
//    typedef GClonesArray<B_Parameters> B_List;
//#else
    typedef TClonesArray B_List;
//WLAV: commented out the following line:
//#endif
    JansEvent();
    void reset();

    JansEventHeader eventHeader;

    // Composite objects
    B_List bList;

    ClassDefOverride(JansEvent, 1);
};


#endif
