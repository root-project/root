#ifndef NTPRECORD
#define NTPRECORD

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// NtpRecord                                                            //
//                                                                      //
// A demo ntuple record                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClonesArray.h"

class NtpShower : public TObject {

public:
  NtpShower() : fEnergy(0) {} // def const'r
  NtpShower(Float_t energy) { fEnergy = energy; } // normal const'r
  virtual ~NtpShower() { this -> Clear(); } 

  Float_t GetEnergy() { return fEnergy; }
  void Clear(Option_t* = "") {}

private:
  Float_t      fEnergy; // shower energy

  ClassDef(NtpShower,1)  //A reconstructed shower 
};

class NtpEvent : public TObject {

public:
  NtpEvent() : fEventNo(-1),fNShower(0),fShwInd(0) {} // def const'r
  NtpEvent(Int_t eventno,Int_t nshower): fEventNo(eventno),
                                         fNShower(nshower),fShwInd(0) {
    if ( fNShower ) {
      fShwInd = new Int_t[fNShower];
      for (Int_t i = 0; i < nshower; i++ ) fShwInd[i] = -1;
    }
  } // normal const'r
  virtual ~NtpEvent() { this -> Clear(); }

  Int_t GetEventNo() { return fEventNo; }
  Int_t GetNShower() { return fNShower; }
  Int_t*  GetShowerIndices() { return fShwInd; }

  void Clear(Option_t* = "") { if ( fShwInd) delete [] fShwInd; fShwInd = 0; }

private:
  Int_t fEventNo; // event number
  Int_t      fNShower; // number of showers in this event
  Int_t*     fShwInd; //[fNShower] array of indices into shower TClonesArray

  ClassDef(NtpEvent,1)  //A reconstructed event 
};

class NtpRecord: public TObject {

 public:

  NtpRecord();
  virtual ~NtpRecord();
  void NtpRecord::Clear(Option_t* option = "");

  TClonesArray* GetShowers() const { return fShowers; }
  TClonesArray* GetEvents() const { return fEvents; }

 private:

  TClonesArray*  fShowers;  //-> Array of showers
  TClonesArray*  fEvents;   //-> Array of events

  static TClonesArray* fgShowers;
  static TClonesArray* fgEvents;

  ClassDef(NtpRecord,1) // A reconstructed spill
};
#endif
