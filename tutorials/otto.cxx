#include "TFile.h"
#include "otto.h"

ClassImp(TUsrHit)
    ClassImp(TUsrHitBuffer)
    ClassImp(TUsrSevtData2)
    ClassImp(TMrbSubevent_Caen)

TClonesArray *TUsrHitBuffer::fgHits = 0;
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
   if(!fgHits)fgHits = new TClonesArray("TUsrHit", fNofEntries);
   fHits = fgHits;
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

void TUsrSevtData2::SetEvent(Int_t ev) {
   Clear();
   fTimeStamp = 100+ev; //in TMrbSubevent_Caen
   fSevtName  = "top";
   fSevtName += ev;
   fMer       = 1000 + ev;
   fPileup    = 2000 + ev;
   for(Int_t i = 1; i <= ev+1; i++) {
      fHitBuffer.AddHit(i);
   }
}
