// @(#)root/eg:$Id$
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
// Is an base class, that defines the interface of ROOT to various    	//
// event generators. Every event generator should inherit from       	//
// TGenerator or its subclasses.                                        //
//                                                                      //
// Derived class can overload the member  function GenerateEvent        //
// to do the actual event generation (e.g., call PYEVNT or similar).    //
//                                                                      //
// The derived class should overload the member function                //
// ImportParticles (both types) to read the internal storage of the     //
// generated event into either the internal TObjArray or the passed     //
// TClonesArray of TParticles.                                          //
//                                                                      //
// If the generator code stores event data in the /HEPEVT/ common block //
// Then the default implementation of ImportParticles should suffice.   //
// The common block /HEPEVT/ is structed like                           //
//                                                                      //
//   /* C */                                                            //
//   typedef struct {                                                   //
//      Int_t    nevhep;                                                //
//      Int_t    nhep;                                                  //
//      Int_t    isthep[4000];                                          //
//      Int_t    idhep[4000];                                           //
//      Int_t    jmohep[4000][2];                                       //
//      Int_t    jdahep[4000][2];                                       //
//      Double_t phep[4000][5];                                         //
//      Double_t vhep[4000][4];                                         //
//   } HEPEVT_DEF;                                                      //
//                                                                      //
//                                                                      //
//   C Fortran                                                          //
//         COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(4000),IDHEP(4000),          //
//       +    JMOHEP(2,4000),JDAHEP(2,4000),PHEP(5,4000),VHEP(4,4000)   //
//         INTEGER NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP               //
//         DOUBLE PRECISION PHEP,VHEP                                   //
//                                                                      //
// The generic member functions SetParameter and GetParameter can be    //
// overloaded to set and get parameters of the event generator.         //
//                                                                      //
// Note, if the derived class interfaces a (set of) Fortran common      //
// blocks (like TPythia, TVenus does), one better make the derived      //
// class a singleton.  That is, something like                          //
//                                                                      //
//     class MyGenerator : public TGenerator                            //
//     {                                                                //
//     public:                                                          //
//       static MyGenerator* Instance()                                 //
//       {                                                              //
//         if (!fgInstance) fgInstance = new MyGenerator;               //
//         return fgInstance;                                           //
//       }                                                              //
//       void  GenerateEvent() { ... }                                  //
//       void  ImportParticles(TClonesArray* a, Option_t opt="") {...}  //
//       Int_t ImportParticles(Option_t opt="") { ... }                 //
//       Int_t    SetParameter(const char* name, Double_t val) { ... }  //
//       Double_t GetParameter(const char* name) { ... }                //
//       virtual ~MyGenerator() { ... }                                 //
//     protected:                                                       //
//       MyGenerator() { ... }                                          //
//       MyGenerator(const MyGenerator& o) { ... }                      //
//       MyGenerator& operator=(const MyGenerator& o) { ... }           //
//       static MyGenerator* fgInstance;                                //
//       ClassDef(MyGenerator,0);                                       //
//     };                                                               //
//                                                                      //
// Having multiple objects accessing the same common blocks is not      //
// safe.                                                                //
//                                                                      //
// concrete TGenerator classes can be loaded in scripts and subseqent-  //
// ly used in compiled code:                                            //
//                                                                      //
//     // MyRun.h                                                       //
//     class MyRun : public TObject                                     //
//     {                                                                //
//     public:                                                          //
//       static MyRun* Instance() { ... }                               //
//       void SetGenerator(TGenerator* g) { fGenerator = g; }           //
//       void Run(Int_t n, Option_t* option="")                         //
//       {                                                              //
//         TFile*        file = TFile::Open("file.root","RECREATE");    //
//         TTree*        tree = new TTree("T","T");                     //
//         TClonesArray* p    = new TClonesArray("TParticles");         //
//         tree->Branch("particles", &p);                               //
//         for (Int_t event = 0; event < n; event++) {                  //
//           fGenerator->GenerateEvent();                               //
//           fGenerator->ImportParticles(p,option);                     //
//           tree->Fill();                                              //
//         }                                                            //
//         file->Write();                                               //
//         file->Close();                                               //
//       }                                                              //
//       ...                                                            //
//     protected:                                                       //
//       TGenerator* fGenerator;                                        //
//       ClassDef(MyRun,0);                                             //
//     };                                                               //
//                                                                      //
//     // Config.C                                                      //
//     void Config()                                                    //
//     {                                                                //
//        MyRun* run = MyRun::Instance();                               //
//        run->SetGenerator(MyGenerator::Instance());                   //
//     }                                                                //
//                                                                      //
//     // main.cxx                                                      //
//     int                                                              //
//     main(int argc, char** argv)                                      //
//     {                                                                //
//       TApplication app("", 0, 0);                                    //
//       gSystem->ProcessLine(".x Config.C");                           //
//       MyRun::Instance()->Run(10);                                    //
//       return 0;                                                      //
//     }                                                                //
//                                                                      //
// This is especially useful for example with TVirtualMC or similar.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGenerator
#define ROOT_TGenerator

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TBrowser;
class TParticle;
class TClonesArray;
class TObjArray;

class TGenerator : public TNamed {

protected:
   Float_t       fPtCut;        //!Pt cut. Do not show primaries below
   Bool_t        fShowNeutrons; //!display neutrons if true
   TObjArray    *fParticles;    //->static container of the primary particles

   TGenerator(const TGenerator& tg) :
   TNamed(tg), fPtCut(tg.fPtCut), fShowNeutrons(tg.fShowNeutrons),fParticles(tg.fParticles) { }
   TGenerator& operator=(const TGenerator& tg) {
      if(this!=&tg) {
         TNamed::operator=(tg); fPtCut=tg.fPtCut; fShowNeutrons=tg.fShowNeutrons;
         fParticles=tg.fParticles;
      }
      return *this;
   }

public:

   TGenerator(): fPtCut(0), fShowNeutrons(kTRUE), fParticles(0) { } //Used by Dictionary
   TGenerator(const char *name, const char *title="Generator class");
   virtual ~TGenerator();
   virtual void            Browse(TBrowser *b);
   virtual Int_t           DistancetoPrimitive(Int_t px, Int_t py);
   virtual void            Draw(Option_t *option="");
   virtual void            ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void            GenerateEvent();
   virtual Double_t        GetParameter(const char* /*name*/) const { return 0.; }
   virtual Int_t           ImportParticles(TClonesArray *particles, Option_t *option="");
   virtual TObjArray      *ImportParticles(Option_t *option="");
   virtual TParticle      *GetParticle(Int_t i) const;
   Int_t                   GetNumberOfParticles() const;
   virtual TObjArray      *GetListOfParticles() const {return fParticles;}
   virtual TObjArray      *GetPrimaries(Option_t *option="") {return ImportParticles(option);}
   Float_t                 GetPtCut() const {return fPtCut;}
   virtual void            Paint(Option_t *option="");
   virtual void            SetParameter(const char* /*name*/,Double_t /*val*/){}
   virtual void            SetPtCut(Float_t ptcut=0); // *MENU*
   virtual void            SetViewRadius(Float_t rbox = 1000); // *MENU*
   virtual void            SetViewRange(Float_t xmin=-10000,Float_t ymin=-10000,Float_t zmin=-10000
                                       ,Float_t xmax=10000,Float_t ymax=10000,Float_t zmax=10000);  // *MENU*
   virtual void            ShowNeutrons(Bool_t show=1); // *MENU*

   ClassDef(TGenerator,1)  //Event generator interface abstract baseclass
};

#endif
