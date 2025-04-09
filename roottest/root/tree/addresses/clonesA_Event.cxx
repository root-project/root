#include "TFile.h"
#include "clonesA_Event.h"

ClassImp(TUsrHit)
    ClassImp(TUsrHitBuffer)
    ClassImp(TUsrSevtData1)
    ClassImp(TUsrSevtData2)
    ClassImp(TMrbSubevent_Caen)

//______________________________________________________
TUsrHit::TUsrHit(Int_t ev) {
   cerr << "ctor TUsrHit " << this << endl;
   fEventNumber = ev;
   fModuleNumber = ev%4;
   fChannel  = ev+1000;
//   for (Int_t i=0;i<3;i++) fEventTime[i] = 100+ev;  
   //fChannelTime = ev * ev;
}

//______________________________________________________

TUsrHitBuffer::TUsrHitBuffer(Int_t maxent) {
   cerr << "ctor TUsrHitBuffer " << this << endl;
   fNofEntries = maxent;
   fNofHits = 0;
   fHits = new TClonesArray("TUsrHit", fNofEntries);
   cerr << "Made clones at ptr " << &fHits << " at addr " << fHits << endl;
}

//______________________________________________________

TUsrHit *TUsrHitBuffer::AddHit(Int_t ev) {
   TClonesArray & hits = *fHits;
   TUsrHit *hit = new(hits[fNofHits++]) TUsrHit(ev);
   return hit;
}

//______________________________________________________

void TUsrHitBuffer::Clear(Option_t *) {
   fHits->Clear();
   fNofHits = 0;
}

//______________________________________________________

void TUsrSevtData1::SetEvent(Int_t ev) {
   Clear();
   cout << "TUsrSevtData1: " << ev << endl;
   fTimeStamp = 100+ev; //in TMrbSubevent_Caen
   fSevtName  = "SubEvent_1_";
   fSevtName += ev;
   fMer       = 1100 + ev;
   fPileup    = 2100 + ev;
   for(Int_t i = 1; i <= ev+1; i++) {
      fHitBuffer.AddHit(i);
   }
   fNiceTrig = -ev;
}
//______________________________________________________

void TUsrSevtData2::SetEvent(Int_t ev) {
   Clear();
   cout << "TUsrSevtData2: " << ev << endl;
   fTimeStamp = 100+ev; //in TMrbSubevent_Caen
   fSevtName  = "SubEvent_2_";
   fSevtName += ev;
   fMer       = 21000 + ev;
   fPileup    = 22000 + ev;
   for(Int_t i = 1; i <= ev+1; i++) {
      fHitBuffer.AddHit(i);
   }
}
