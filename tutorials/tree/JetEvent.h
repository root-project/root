#ifndef ROOT_JetEvent
#define ROOT_JetEvent

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// JetEvent                                                             //
//                                                                      //
// Description of the event and track parameters                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClonesArray.h"
#include "TRefArray.h"
#include "TVector3.h"

class Hit : public TObject {

public:
   Float_t      fX;           //X of hit
   Float_t      fY;           //Y of hit
   Float_t      fZ;           //Z of hit

public:
   Hit() { }
   virtual ~Hit() { }

   ClassDef(Hit,1)  //A track hit
};

class Track : public TObject {

public:
   Float_t      fPx;           //X component of the momentum
   Float_t      fPy;           //Y component of the momentum
   Float_t      fPz;           //Z component of the momentum
   Int_t        fNhit;         //Number of hits for this track
   TRefArray    fHits;         //List of Hits for this track

public:
   Track() { }
   virtual ~Track() { }
   Int_t         GetNhit() const { return fNhit; }
   TRefArray   &GetHits()  {return fHits; }

   ClassDef(Track,1)  //A track segment
};


class Jet : public TObject {

public:
   Double_t   fPt;       //Pt of jet
   Double_t   fPhi;      //Phi of jet
   TRefArray  fTracks;   //List of tracks in the jet

public:
   Jet() { }
   virtual ~Jet(){ }
   TRefArray   &GetTracks() {return fTracks; }

   ClassDef(Jet,1)  //Jet class
};

class JetEvent : public TObject {

private:
   TVector3       fVertex;            //vertex coordinates
   Int_t          fNjet;              //Number of jets
   Int_t          fNtrack;            //Number of tracks
   Int_t          fNhitA;             //Number of hist in detector A
   Int_t          fNhitB;             //Number of hist in detector B
   TClonesArray  *fJets;              //->array with all jets
   TClonesArray  *fTracks;            //->array with all tracks
   TClonesArray  *fHitsA;             //->array of hits in detector A
   TClonesArray  *fHitsB;             //->array of hits in detector B

   static TClonesArray *fgJets;
   static TClonesArray *fgTracks;
   static TClonesArray *fgHitsA;
   static TClonesArray *fgHitsB;

public:
   JetEvent();
   virtual ~JetEvent();
   void          Build(Int_t jetm=3, Int_t trackm=10, Int_t hitam=100, Int_t hitbm=10);
   void          Clear(Option_t *option ="");
   void          Reset(Option_t *option ="");
   Int_t         GetNjet()   const { return fNjet; }
   Int_t         GetNtrack() const { return fNtrack; }
   Int_t         GetNhitA()  const { return fNhitA; }
   Int_t         GetNhitB()  const { return fNhitB; }
   Jet          *AddJet();
   Track        *AddTrack();
   Hit          *AddHitA();
   Hit          *AddHitB();
   TClonesArray *GetJets() const { return fJets; }

   ClassDef(JetEvent,1)  //Event structure
};

#endif




















