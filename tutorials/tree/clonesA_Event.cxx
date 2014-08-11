#include "TFile.h"
#include "TClonesArray.h"
#include "clonesA_Event.h"

ClassImp(TUsrHit)
    ClassImp(TUsrHitBuffer)
    ClassImp(TUsrSevtData1)
    ClassImp(TUsrSevtData2)
    ClassImp(TMrbSubevent_Caen)

//______________________________________________________
   TUsrHit::TUsrHit(Int_t ev) {
   fEventNumber = ev;
   fModuleNumber = ev%4;
   fChannel  = ev+1000;
   for (Int_t i=0;i<3;i++) fEventTime[i] = 100+ev;
}

//______________________________________________________

TUsrHitBuffer::TUsrHitBuffer(Int_t maxent) {
   fNofEntries = maxent;
   fNofHits = 0;
   fHits = new TClonesArray("TUsrHit", fNofEntries);
   std::cout << "ctor TUsrHitBuffer " << this << std::endl;
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
   std::cout << "TUsrSevtData1: " << ev << std::endl;
   fTimeStamp = 100+ev; //in TMrbSubevent_Caen
   fSevtName  = "SubEvent_1_";
   fSevtName += ev;
   fMer       = 1100 + ev;
   fPileup    = 2100 + ev;
   for(Int_t i = 1; i <= ev+1; i++) {
      fHitBuffer.AddHit(i);
   }
}
//______________________________________________________

void TUsrSevtData2::SetEvent(Int_t ev) {
   Clear();
   std::cout << "TUsrSevtData2: " << ev << std::endl;
   fTimeStamp = 100+ev; //in TMrbSubevent_Caen
   fSevtName  = "SubEvent_2_";
   fSevtName += ev;
   fMer       = 21000 + ev;
   fPileup    = 22000 + ev;
   for(Int_t i = 1; i <= ev+1; i++) {
      fHitBuffer.AddHit(i);
   }
}
