/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   AR, Aaron Roodman, Stanford University, roodman@slac.stanford.edu 
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   13-Dec-1999 AR Created initial version
 *   20-Mar-2000 AR Make methods const, add _DeltaZScale and _DeltaZSignFlip
 *   28-Mar-2001 WV Migrate to RooFitCore and rename from RooBlind to RooBlindTools
 *                  Convert basic types to ROOT names
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

//-----------------------
// This Class's Header --
//-----------------------
#include "RooFitCore/RooBlindTools.hh"

#include <iostream.h>
#include <fstream.h>
#include <math.h>  
#include <string.h>
#include <ctype.h>

ClassImp(RooBlindTools)

//----------------
// Constructors --
//----------------
RooBlindTools::RooBlindTools(const char *stSeed, blindMode Mode):
  _mode(Mode)
{
  setup(stSeed);
}

RooBlindTools::RooBlindTools(const RooBlindTools& blindTool):
  _mode(blindTool.mode())
{
  setup(blindTool.stSeed());
}

void RooBlindTools::setup(const char *stSeed)
{

  _stSeed = stSeed;

  _DeltaZScale = 1.56;

  _DeltaZOffset = _DeltaZScale*MakeOffset("abcdefghijklmnopqrstuvwxyz");

  _DeltaZSignFlip = MakeSignFlip("ijklmnopqrstuvwxyzabcdefgh");

  _AsymOffset = MakeGaussianOffset("opqrstuvwxyzabcdefghijklmn");

  _AsymSignFlip = MakeSignFlip("zyxwvutsrqponmlkjihgfedcba");

  _DeltaMScale = 0.1;

  _DeltaMOffset = _DeltaMScale*MakeOffset("opqrstuvwxyzabcdefghijklmn");

  _MysteryPhase = 3.14159 * 
                  MakeOffset("wxyzabcdefghijklmnopqrstuv");

  _STagConstant = Randomizer("fghijklmnopqrstuvwxyzabcde");

}


//--------------
// Destructor --
//--------------
RooBlindTools::~RooBlindTools(){}

//-------------
// Functions --
//-------------


Double_t RooBlindTools::HideDeltaZ(Double_t DeltaZ, Double_t STag)const{

  Int_t sTag = SignOfTag(STag);
  Double_t DeltaZPrime = _DeltaZSignFlip*DeltaZ*sTag + _DeltaZOffset;

  return DeltaZPrime;
}

Double_t RooBlindTools::HiDelZPdG(Double_t DeltaZ, Double_t STag, Double_t PdG) const{

  Int_t sTag = SignOfTag(STag);
  Double_t DeltaZPrime = _DeltaZSignFlip*(DeltaZ - PdG)*sTag + _DeltaZOffset;

  return DeltaZPrime;
}

Double_t RooBlindTools::UnHideDeltaZ(Double_t DeltaZPrime, Double_t STag) const{

  Int_t sTag = SignOfTag(STag);
  Double_t DeltaZ = (DeltaZPrime - _DeltaZOffset)/(sTag*_DeltaZSignFlip);

  return DeltaZ;
}

Double_t RooBlindTools::UnHiDelZPdG(Double_t DeltaZPrime, Double_t STag, Double_t PdG) const{

  Int_t sTag = SignOfTag(STag);
  Double_t DeltaZ = PdG + (DeltaZPrime - _DeltaZOffset)/(sTag*_DeltaZSignFlip);

  return DeltaZ;
}

Double_t RooBlindTools::UnHideAsym(Double_t AsymPrime) const{

  if(mode()==dataonly) return AsymPrime;

  Double_t Asym = (AsymPrime - _AsymOffset)/_AsymSignFlip;

  return Asym;
}

Double_t RooBlindTools::HideAsym(Double_t Asym) const{

  if(mode()==dataonly) return Asym;

  Double_t AsymPrime = Asym*_AsymSignFlip + _AsymOffset;

  return AsymPrime;
}


Double_t RooBlindTools::UnHideDeltaM(Double_t DeltaMPrime) const{

  if(mode()==dataonly) return DeltaMPrime;

  Double_t DeltaM = DeltaMPrime - _DeltaMOffset;

  return DeltaM;
}

Double_t RooBlindTools::HideDeltaM(Double_t DeltaM) const{

  if(mode()==dataonly) return DeltaM;

  Double_t DeltaMPrime = DeltaM + _DeltaMOffset;

  return DeltaMPrime;
}

Double_t RooBlindTools::UnHiAsPdG(Double_t AsymPrime, Double_t PdG) const{

  if(mode()==dataonly) return AsymPrime;

  Double_t Asym = PdG + (AsymPrime - _AsymOffset)/_AsymSignFlip;

  return Asym;
}

Double_t RooBlindTools::MysteryPhase() const{

  if(mode()==dataonly) return 0.0;

  return _MysteryPhase;
}

Double_t RooBlindTools::HiAsPdG(Double_t Asym, Double_t PdG) const{

  if(mode()==dataonly) return Asym;

  Double_t AsymPrime = (Asym - PdG)*_AsymSignFlip + _AsymOffset;

  return AsymPrime;
}


Double_t RooBlindTools::RandomizeTag(Double_t STag, Int_t EventNumber) const{

  Int_t Seed = EventNumber % 7997 + 2; 
  Double_t r = PseudoRandom(Seed);
  Double_t STagPrime(0.0);

  if (r < _STagConstant){
    STagPrime = STag;
  } else {
    STagPrime = -1.0 * STag ;
  }

  return STagPrime;

}
   

Double_t RooBlindTools::Randomizer(char *StringAlphabet) const{

  char lowerseed[1024] ;
  strcpy(lowerseed,_stSeed) ;

  Int_t lengthSeed = strlen(lowerseed);

  for (Int_t j=0; j<lengthSeed; j++){
    lowerseed[j] =tolower(_stSeed[j]);
  }                                                                                                                                                                        
  Int_t sumSeed = 0;
  for (Int_t i=0; i<lengthSeed; i++){
    for (Int_t iAlphabet=0; iAlphabet<26; iAlphabet++){
      if ( lowerseed[i] == StringAlphabet[iAlphabet] ){
	sumSeed = sumSeed + iAlphabet ;
      }
    }      
  }

  if (sumSeed<1 || sumSeed>8000 || lengthSeed<5) {
    cout<< "RooBlindTools::Randomizer: Your String Seed is Bad" <<endl;
  }
  
  Int_t ia = 8121;
  Int_t ic = 28411;
  Int_t im = 134456;
  UInt_t jRan = (sumSeed*ia + ic) % im;

  jRan = (jRan*ia + ic) % im;
  jRan = (jRan*ia + ic) % im;
  jRan = (jRan*ia + ic) % im;

  Double_t theRan = (float) jRan / (float) im;     

  return theRan;    //theRan is between 0.0 - 1.0

}

Double_t RooBlindTools::PseudoRandom(Int_t Seed) const{

  if (Seed<1 || Seed>8000 ) {
    cout<< "RooBlindTools::PseudoRandom: Your integer Seed is Bad" <<endl;
  }
 
  Int_t ia = 8121;
  Int_t ic = 28411;
  Int_t im = 134456;
  UInt_t jRan = (Seed*ia + ic) % im;

  jRan = (jRan*ia + ic) % im;
  jRan = (jRan*ia + ic) % im;
  jRan = (jRan*ia + ic) % im;

  Double_t theRan = (float) jRan / (float) im;     

  return theRan;    //theRan is between 0.0 - 1.0

}



Double_t RooBlindTools::MakeOffset(char *StringAlphabet) const{

  Double_t theRan = Randomizer(StringAlphabet);

  Double_t theOffset = (2.0)*theRan - (1.0);

  return theOffset;   //theOffset lies between -1.0 and 1.0
}



Double_t RooBlindTools::MakeGaussianOffset(char *StringAlphabet) const{

  Double_t theRan1 = Randomizer(StringAlphabet);
  Double_t theRan2 = Randomizer("cdefghijklmnopqrstuvwxyzab");

  if (theRan1==0.0 || theRan1==1.0){
    theRan1 = 0.5;
  }
  if (theRan2==0.0 || theRan2==1.0){
    theRan2 = 0.5;
  }

  Double_t theOffset = sin(2.0*3.14159*theRan1)*sqrt(-2.0*log(theRan2));

  return theOffset;   //theOffset is Gaussian with mean 0, sigma 1
}


Double_t RooBlindTools::MakeSignFlip(char *StringAlphabet) const{

  Double_t theRan = Randomizer(StringAlphabet);

  Double_t theSignFlip = 1.0;
  if (theRan>0.5){
    theSignFlip = 1.0;
  } else {
    theSignFlip = -1.0;
  }

  return theSignFlip;  //theSignFlip is = +1 or -1
}

Int_t RooBlindTools::SignOfTag(Double_t STag) const{

  Int_t sTag;
  if (STag < 0.0){
    sTag = -1;
  } 
  else if (STag > 0.0) {
    sTag = 1;
  }
  else {
    sTag = 1;
  }

  return sTag;
   
}

 
