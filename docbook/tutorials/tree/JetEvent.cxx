// A JetEvent emulates 2 detectors A and B producing each
// a TClonesArray of Hit objects.
// A TClonesArray  of Track objects is built with Hits objects
// of detectors A and B. Eack Track object has a TRefArray of hits.
// A TClonesArray of Jets is made with a subset of the Track objects
// also stored in a TRefArray.
// see $ROOTSYS/tutorials/jets.C for an example creating a Tree
// with JetEvents.
      
#include "TMath.h"   
#include "TRandom.h"   
#include "JetEvent.h"

ClassImp(Jet)
ClassImp(Track)
ClassImp(Hit)
ClassImp(JetEvent)

TClonesArray *JetEvent::fgJets   = 0;
TClonesArray *JetEvent::fgTracks = 0;
TClonesArray *JetEvent::fgHitsA  = 0;
TClonesArray *JetEvent::fgHitsB  = 0;

//______________________________________________________________________________
JetEvent::JetEvent()
{
   // Create a JetEvent object.
   // When the constructor is invoked for the first time, the class static
   // variables fgxxx are 0 and the TClonesArray fgxxx are created.

   if (!fgTracks) fgTracks = new TClonesArray("Track", 100);
   if (!fgJets)   fgJets   = new TClonesArray("Jet", 10);
   if (!fgHitsA)  fgHitsA  = new TClonesArray("Hit", 10000);
   if (!fgHitsB)  fgHitsB  = new TClonesArray("Hit", 1000);
   fJets   = fgJets;
   fTracks = fgTracks;
   fHitsA  = fgHitsA;
   fHitsB  = fgHitsB;
}

//______________________________________________________________________________
JetEvent::~JetEvent()
{
   Reset();
}

//______________________________________________________________________________
void JetEvent::Build(Int_t jetm, Int_t trackm, Int_t hitam, Int_t hitbm) {
   //Build one event
   
   //Save current Object count
   Int_t ObjectNumber = TProcessID::GetObjectCount();
   Clear();

   Hit *hit;
   Track *track;
   Jet *jet;
   fNjet   = fNtrack = fNhitA  = fNhitB  = 0;
   
   fVertex.SetXYZ(gRandom->Gaus(0,0.1),
                  gRandom->Gaus(0,0.2),
                  gRandom->Gaus(0,10));
      
   Int_t njets = (Int_t)gRandom->Gaus(jetm,1); if (njets < 1) njets = 1;
   for (Int_t j=0;j<njets;j++) {
      jet = AddJet();
      jet->fPt = gRandom->Gaus(0,10);
      jet->fPhi = 2*TMath::Pi()*gRandom->Rndm();
      Int_t ntracks = (Int_t)gRandom->Gaus(trackm,3); if (ntracks < 1) ntracks = 1;
      for (Int_t t=0;t<ntracks;t++) {
         track = AddTrack();
         track->fPx = gRandom->Gaus(0,1);
         track->fPy = gRandom->Gaus(0,1);
         track->fPz = gRandom->Gaus(0,5);
	 jet->fTracks.Add(track);
         Int_t nhitsA = (Int_t)gRandom->Gaus(hitam,5);
         for (Int_t ha=0;ha<nhitsA;ha++) {
            hit = AddHitA();
            hit->fX = 10000*j + 100*t +ha;
            hit->fY = 10000*j + 100*t +ha+0.1;
            hit->fZ = 10000*j + 100*t +ha+0.2;
	    track->fHits.Add(hit);
         }
         Int_t nhitsB = (Int_t)gRandom->Gaus(hitbm,2);
         for (Int_t hb=0;hb<nhitsB;hb++) {
            hit = AddHitB();
            hit->fX = 20000*j + 100*t +hb+0.3;
            hit->fY = 20000*j + 100*t +hb+0.4;
            hit->fZ = 20000*j + 100*t +hb+0.5;
	    track->fHits.Add(hit);
         }
         track->fNhit = nhitsA + nhitsB;
      }
   }         
  //Restore Object count 
  //To save space in the table keeping track of all referenced objects
  //we assume that our events do not address each other. We reset the 
  //object count to what it was at the beginning of the event.
  TProcessID::SetObjectCount(ObjectNumber);
}


//______________________________________________________________________________
Jet *JetEvent::AddJet()
{
   // Add a new Jet to the list of tracks for this event.

   TClonesArray &jets = *fJets;
   Jet *jet = new(jets[fNjet++]) Jet();
   return jet;
}


//______________________________________________________________________________
Track *JetEvent::AddTrack()
{
   // Add a new track to the list of tracks for this event.

   TClonesArray &tracks = *fTracks;
   Track *track = new(tracks[fNtrack++]) Track();
   return track;
}


//______________________________________________________________________________
Hit *JetEvent::AddHitA()
{
   // Add a new hit to the list of hits in detector A

   TClonesArray &hitsA = *fHitsA;
   Hit *hit = new(hitsA[fNhitA++]) Hit();
   return hit;
}

//______________________________________________________________________________
Hit *JetEvent::AddHitB()
{
   // Add a new hit to the list of hits in detector B

   TClonesArray &hitsB = *fHitsB;
   Hit *hit = new(hitsB[fNhitB++]) Hit();
   return hit;
}

//______________________________________________________________________________
void JetEvent::Clear(Option_t *option)
{
   fJets->Clear(option);
   fTracks->Clear(option);
   fHitsA->Clear(option);
   fHitsB->Clear(option);
}

//______________________________________________________________________________
void JetEvent::Reset(Option_t *)
{
// Static function to reset all static objects for this event

   delete fgJets;   fgJets = 0;
   delete fgTracks; fgTracks = 0;
   delete fgHitsA;  fgHitsA = 0;
   delete fgHitsB;  fgHitsB = 0;
}






