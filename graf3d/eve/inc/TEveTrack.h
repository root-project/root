// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTrack
#define ROOT_TEveTrack

#include <vector>

#include "TEveVSDStructs.h"
#include "TEveElement.h"
#include "TEveLine.h"

#include "TPolyMarker3D.h"
#include "TMarker.h"

class TEveTrackPropagator;
class TEveTrackList;

class TEveTrack : public TEveLine
{
   friend class TEveTrackPropagator;
   friend class TEveTrackList;
   friend class TEveTrackCounter;
   friend class TEveTrackGL;

private:
   TEveTrack& operator=(const TEveTrack&); // Not implemented

public:
   typedef std::vector<TEvePathMark>    vPathMark_t;
   typedef vPathMark_t::iterator        vPathMark_i;
   typedef vPathMark_t::const_iterator  vPathMark_ci;

protected:
   TEveVector         fV;          // Starting vertex
   TEveVector         fP;          // Starting momentum
   TEveVector         fPEnd;       // Momentum at the last point of extrapolation
   Double_t           fBeta;       // Relativistic beta factor
   Int_t              fPdg;        // PDG code
   Int_t              fCharge;     // Charge in units of e0
   Int_t              fLabel;      // Simulation label
   Int_t              fIndex;      // Reconstruction index
   vPathMark_t        fPathMarks;  // TEveVector of known points along the track

   TEveTrackPropagator *fPropagator;   // Pointer to shared render-style

public:
   TEveTrack();
   TEveTrack(TParticle* t, Int_t label, TEveTrackPropagator* rs);
   TEveTrack(TEveMCTrack*  t, TEveTrackPropagator* rs);
   TEveTrack(TEveRecTrack* t, TEveTrackPropagator* rs);
   TEveTrack(const TEveTrack& t);
   virtual ~TEveTrack();

   virtual void SetStdTitle();

   virtual void SetTrackParams(const TEveTrack& t);
   virtual void SetPathMarks  (const TEveTrack& t);

   virtual void MakeTrack(Bool_t recurse=kTRUE);

   TEveTrackPropagator* GetPropagator() const  { return fPropagator; }
   void SetPropagator(TEveTrackPropagator* rs);
   void SetAttLineAttMarker(TEveTrackList* tl);

   const TEveVector& GetVertex()      const { return fV;    }
   const TEveVector& GetMomentum()    const { return fP;    }
   const TEveVector& GetEndMomentum() const { return fPEnd; }

   Int_t GetPdg()    const   { return fPdg;   }
   void SetPdg(Int_t pdg)    { fPdg = pdg;    }
   Int_t GetCharge() const   { return fCharge; }
   void SetCharge(Int_t chg) { fCharge = chg; }
   Int_t GetLabel()  const   { return fLabel; }
   void  SetLabel(Int_t lbl) { fLabel = lbl;  }
   Int_t GetIndex()  const   { return fIndex; }
   void  SetIndex(Int_t idx) { fIndex = idx;  }

   void  AddPathMark(const TEvePathMark& pm) { fPathMarks.push_back(pm); }
   void  SortPathMarksByTime();
         vPathMark_t& RefPathMarks()       { return fPathMarks; }
   const vPathMark_t& RefPathMarks() const { return fPathMarks; }

   //--------------------------------

   void ImportHits();              // *MENU*
   void ImportClusters();          // *MENU*
   void ImportClustersFromIndex(); // *MENU*
   void ImportKine();              // *MENU*
   void ImportKineWithArgs(Bool_t importMother=kTRUE, Bool_t impDaugters=kTRUE,
                           Bool_t colorPdg    =kTRUE, Bool_t recurse    =kTRUE); // *MENU*
   void PrintKineStack();          // *MENU*
   void PrintPathMarks();          // *MENU*

   //--------------------------------

   virtual void SecSelected(TEveTrack*); // *SIGNAL*
   virtual void SetLineStyle(Style_t lstyle);

   virtual const TGPicture* GetListTreeIcon(Bool_t open=kFALSE);

   virtual void CopyVizParams(const TEveElement* el);

   virtual TClass* ProjectedClass() const;

   ClassDef(TEveTrack, 1); // Track with given vertex, momentum and optional referece-points (path-marks) along its path.
};

/******************************************************************************/
// TEveTrackList
/******************************************************************************/

class TEveTrackList : public TEveElementList,
		      public TEveProjectable,
		      public TAttMarker,
		      public TAttLine
{
   friend class TEveTrackListEditor;

private:
   TEveTrackList(const TEveTrackList&);            // Not implemented
   TEveTrackList& operator=(const TEveTrackList&); // Not implemented

protected:
   TEveTrackPropagator* fPropagator;   // Basic track rendering parameters, not enforced to elements.

   Bool_t               fRecurse;    // Recurse when propagating marker/line/etc attributes to tracks.

   Bool_t               fRnrLine;    // Render track as line.
   Bool_t               fRnrPoints;  // Render track as points.

   Float_t              fMinPt;      // Minimum track pT for display selection.
   Float_t              fMaxPt;      // Maximum track pT for display selection.
   Float_t              fLimPt;      // Highest track pT in the container.
   Float_t              fMinP;       // Minimum track p for display selection.
   Float_t              fMaxP;       // Maximum track p for display selection.
   Float_t              fLimP;       // Highest track p in the container.

   Float_t RoundMomentumLimit(Float_t x);

public:
   TEveTrackList(TEveTrackPropagator* rs=0);
   TEveTrackList(const Text_t* name, TEveTrackPropagator* rs=0);
   virtual ~TEveTrackList();

   void  MakeTracks(Bool_t recurse=kTRUE);
   void  FindMomentumLimits(TEveElement* el, Bool_t recurse);

   void  SetPropagator(TEveTrackPropagator* rs);
   TEveTrackPropagator* GetPropagator() { return fPropagator; }

   //--------------------------------

   virtual void   SetMainColor(Color_t c);
   virtual void   SetLineColor(Color_t c) { SetMainColor(c); }
   virtual void   SetLineColor(Color_t c, TEveElement* el);
   virtual void   SetLineWidth(Width_t w);
   virtual void   SetLineWidth(Width_t w, TEveElement* el);
   virtual void   SetLineStyle(Style_t s);
   virtual void   SetLineStyle(Style_t s, TEveElement* el);

   virtual void   SetMarkerColor(Color_t c);
   virtual void   SetMarkerColor(Color_t c, TEveElement* el);
   virtual void   SetMarkerSize(Size_t s);
   virtual void   SetMarkerSize(Size_t s, TEveElement* el);
   virtual void   SetMarkerStyle(Style_t s);
   virtual void   SetMarkerStyle(Style_t s, TEveElement* el);

   void SetRnrLine(Bool_t rnr);
   void SetRnrLine(Bool_t rnr, TEveElement* el);
   Bool_t GetRnrLine(){return fRnrLine;}

   void SetRnrPoints(Bool_t r);
   void SetRnrPoints(Bool_t r, TEveElement* el);
   Bool_t GetRnrPoints(){return fRnrPoints;}

   void SelectByPt(Float_t min_pt, Float_t max_pt);
   void SelectByPt(Float_t min_pt, Float_t max_pt, TEveElement* el);
   void SelectByP (Float_t min_p,  Float_t max_p);
   void SelectByP (Float_t min_p,  Float_t max_p,  TEveElement* el);

   //--------------------------------

   TEveTrack* FindTrackByLabel(Int_t label); // *MENU*
   TEveTrack* FindTrackByIndex(Int_t index); // *MENU*

   void ImportHits();     // *MENU*
   void ImportClusters(); // *MENU*

   virtual void CopyVizParams(const TEveElement* el);

   virtual TClass* ProjectedClass() const;

   ClassDef(TEveTrackList, 1); // A list of tracks supporting change of common attributes and selection based on track parameters.
};


/******************************************************************************/
// TEveTrackCounter
/******************************************************************************/

class TEveTrackCounter : public TEveElement, public TNamed
{
   friend class TEveTrackCounterEditor;

public:
   enum EClickAction_e { kCA_PrintTrackInfo, kCA_ToggleTrack };

private:
   TEveTrackCounter(const TEveTrackCounter&);            // Not implemented
   TEveTrackCounter& operator=(const TEveTrackCounter&); // Not implemented

protected:
   Int_t fBadLineStyle;  // TEveLine-style used for secondary/bad tracks.
   Int_t fClickAction;   // Action to take when a track is ctrl-clicked.

   Int_t fEventId;       // Current event-id.

   Int_t fAllTracks;     // Counter of all tracks.
   Int_t fGoodTracks;    // Counter of good tracks.

   TList fTrackLists;    // List of TrackLists registered for management.

public:
   TEveTrackCounter(const Text_t* name="TEveTrackCounter", const Text_t* title="");
   virtual ~TEveTrackCounter();

   Int_t GetEventId() const { return fEventId; }
   void  SetEventId(Int_t id) { fEventId = id; }

   void Reset();

   void RegisterTracks(TEveTrackList* tlist, Bool_t goodTracks);

   void DoTrackAction(TEveTrack* track);

   Int_t GetClickAction() const  { return fClickAction; }
   void  SetClickAction(Int_t a) { fClickAction = a; }

   void OutputEventTracks(FILE* out=0);

   static TEveTrackCounter* fgInstance;

   ClassDef(TEveTrackCounter, 1); // Class for selection of good/primary tracks with basic processing functionality.
};


#endif
