// @(#)root/eg:$Name$:$Id$
// Author: Ola Nordmann   21/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGenerator                                                           //
//                                                                      //
// Is an abstact base class, that defines the interface of ROOT and the //
// various event generators. Every event generator should inherit from  //
// TGenerator or its subclasses.                                        //
//                                                                      //
// Every class inherited from TGenerator knows already the interface to //
// the /HEPEVT/ common block. So in the event creation of the various   //
// generators, the /HEPEVT/ common block should be filled               //
// The ImportParticles method then parses the result from the event     //
// generators into a TClonesArray of TParticle objects.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGenerator
#define ROOT_TGenerator

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TClonesArray
#include "TClonesArray.h"
#endif

class TBrowser;
class TParticle;

class TGenerator : public TNamed {
 protected:
        Float_t       fPtCut;        //Pt cut. Do not show primaries below
        Bool_t        fShowNeutrons; //display neutrons if true
        TObjArray    *fParticles;    //static container of the primary particles
 public:

        TGenerator(){}; //Used by Dictionary
        TGenerator(const char *name, const char *title="Generator class");
        virtual ~TGenerator();
        virtual void            Browse(TBrowser *b);
        virtual Int_t           DistancetoPrimitive(Int_t px, Int_t py);
        virtual void            Draw(Option_t *option="");
        virtual void            ExecuteEvent(Int_t event, Int_t px, Int_t py);
        virtual Int_t           ImportParticles(TClonesArray *particles, Option_t *option="");
        virtual TObjArray      *ImportParticles(Option_t *option="");
        virtual TParticle      *GetParticle(Int_t i);
        Int_t                   GetNumberOfParticles() {return fParticles->GetLast()+1;}
        virtual TObjArray      *GetListOfParticles() {return fParticles;}
        virtual TObjArray      *GetPrimaries(Option_t *option="") {return ImportParticles(option);}
        Float_t                 GetPtCut() {return fPtCut;}
        virtual void            Paint(Option_t *option="");
        virtual void            SetPtCut(Float_t ptcut=0); // *MENU*
        virtual void            SetViewRadius(Float_t rbox = 1000); //*MENU*
        virtual void            SetViewRange(Float_t xmin=-10000,Float_t ymin=-10000,Float_t zmin=-10000
                                            ,Float_t xmax=10000,Float_t ymax=10000,Float_t zmax=10000);  // *MENU*
        virtual void            ShowNeutrons(Bool_t show=1); // *MENU*

        ClassDef(TGenerator,1)  //Event generator interface abstract baseclass
};

#endif
