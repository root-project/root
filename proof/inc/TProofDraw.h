// @(#)root/proof:$Name:  $:$Id: TProofDraw.h,v 1.7 2005/03/18 22:41:27 rdm Exp $
// Author: Maarten Ballintijn   24/09/2003

#ifndef ROOT_TProofDraw
#define ROOT_TProofDraw


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofDraw                                                           //
//                                                                      //
// Implement Tree drawing using PROOF.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TSelector
#include "TSelector.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TTreeDrawArgsParser
#include "TTreeDrawArgsParser.h"
#endif

class TTree;
class TTreeFormulaManager;
class TTreeFormula;
class TStatus;
class TH1;
class TEventList;
class TProfile;
class TProfile2D;
class TProofVarArray;
class TGraph;
class TPolyMarker3D;

class TProofDraw : public TSelector {

protected:
   TTreeDrawArgsParser  fTreeDrawArgsParser;
   TStatus             *fStatus;
   TString              fSelection;
   TString              fInitialExp;
   TTreeFormulaManager *fManager;
   TTree               *fTree;
   TTreeFormula        *fVar[4];         //  Pointer to variable formula
   TTreeFormula        *fSelect;         //  Pointer to selection formula
   Int_t                fMultiplicity;   //  Indicator of the variability of the size of entries
   Bool_t               fObjEval;        //  true if fVar1 returns an object (or pointer to).
   Int_t                fDimension;      //  Dimension of the current expression

   void     SetError(const char *sub, const char *mesg);

protected:
   enum { kWarn = BIT(12) };

   virtual Bool_t      CompileVariables();
   virtual void        ClearFormula();
   virtual Bool_t      ProcessSingle(Long64_t /*entry*/, Int_t /*i*/);
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v) = 0;

public:
   TProofDraw();
   virtual            ~TProofDraw();
   virtual int         Version() const { return 1; }
   virtual void        Init(TTree *);
   virtual void        Begin(TTree *);
   virtual void        SlaveBegin(TTree *);
   virtual Bool_t      Notify();
   virtual Bool_t      Process(Long64_t /*entry*/);
   virtual void        SlaveTerminate();
   virtual void        Terminate();

   ClassDef(TProofDraw,0)  //Tree drawing selector for PROOF
};


class TProofDrawHist : public TProofDraw {

protected:
   TH1                 *fHistogram;

   virtual void        Begin1D(TTree *t);
   virtual void        Begin2D(TTree *t);
   virtual void        Begin3D(TTree *t);
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v);

public:
   TProofDrawHist() : fHistogram(0) { }
   virtual void        Begin(TTree *t);
   virtual void        Init(TTree *);
   virtual void        SlaveBegin(TTree *);
   virtual void        Terminate();

   ClassDef(TProofDrawHist,0)  //Tree drawing selector for PROOF
};


class TProofDrawEventList : public TProofDraw {

protected:
    TEventList*    fElist;          //  event list
    TList*         fEventLists;     //  a list of EventLists
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v);

public:
   TProofDrawEventList() : fElist(0), fEventLists(0) {}
   ~TProofDrawEventList();
   virtual void        Init(TTree *);
   virtual void        SlaveBegin(TTree *);
   virtual void        SlaveTerminate();
   virtual void        Terminate();

   ClassDef(TProofDrawEventList,0)  //Tree drawing selector for PROOF
};


class TProofDrawProfile : public TProofDraw {

protected:
   TProfile           *fProfile;
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v);

public:
   TProofDrawProfile() : fProfile(0) { }
   virtual void        Init(TTree *);
   virtual void        Begin(TTree *t);
   virtual void        SlaveBegin(TTree *);
   virtual void        Terminate();

   ClassDef(TProofDrawProfile,0)  //Tree drawing selector for PROOF
};


class TProofDrawProfile2D : public TProofDraw {

protected:
   TProfile2D           *fProfile;
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v);

public:
   TProofDrawProfile2D() : fProfile(0) { }
   virtual void        Init(TTree *);
   virtual void        Begin(TTree *t);
   virtual void        SlaveBegin(TTree *);
   virtual void        Terminate();

   ClassDef(TProofDrawProfile2D,0)  //Tree drawing selector for PROOF
};


class TProofDrawGraph : public TProofDraw {

protected:
   TGraph             *fGraph;
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v);

public:
   TProofDrawGraph() : fGraph(0) { }
   virtual void        Init(TTree *tree);
   virtual void        SlaveBegin(TTree *);
   virtual void        Terminate();

   ClassDef(TProofDrawGraph,0)  //Tree drawing selector for PROOF
};


class TProofDrawPolyMarker3D : public TProofDraw {

protected:
   TPolyMarker3D      *fPolyMarker3D;
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v);

public:
   TProofDrawPolyMarker3D() : fPolyMarker3D(0) { }
   virtual void        Init(TTree *tree);
   virtual void        SlaveBegin(TTree *);
   virtual void        Terminate();

   ClassDef(TProofDrawPolyMarker3D,0)  //Tree drawing selector for PROOF
};


class TProofDrawListOfGraphs : public TProofDraw {

protected:
   TProofVarArray     *fScatterPlot;
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v);

public:
   TProofDrawListOfGraphs() : fScatterPlot(0) { }
   virtual void        SlaveBegin(TTree *);
   virtual void        Terminate();

   ClassDef(TProofDrawListOfGraphs,0)  //Tree drawing selector for PROOF
};


class TProofDrawListOfPolyMarkers3D : public TProofDraw {

protected:
   TProofVarArray       *fScatterPlot;
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v);

public:
   TProofDrawListOfPolyMarkers3D() : fScatterPlot(0) { }
   virtual void        SlaveBegin(TTree *);
   virtual void        Terminate();

   ClassDef(TProofDrawListOfPolyMarkers3D,0)  //Tree drawing selector for PROOF
};

#endif
