// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn   24/09/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofDraw
#define ROOT_TProofDraw


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofDraw                                                           //
//                                                                      //
// Implement Tree drawing using PROOF.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSelector.h"

#include "TString.h"

#include "TTreeDrawArgsParser.h"

#include "TNamed.h"

#include <vector>


class TTree;
class TTreeFormulaManager;
class TTreeFormula;
class TStatus;
class TH1;
class TEventList;
class TEntryList;
class TProfile;
class TProfile2D;
class TGraph;
class TPolyMarker3D;
class TCollection;


class TProofDraw : public TSelector {

friend class TProofPlayer;

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
   Double_t             fWeight;         //  Global weight for fill actions

   void     FillWeight();
   void     SetCanvas(const char *objname);
   void     SetDrawAtt(TObject *o);
   void     SetError(const char *sub, const char *mesg);

protected:
   enum { kWarn = BIT(12) };

   virtual Bool_t      CompileVariables();
   virtual void        ClearFormula();
   virtual Bool_t      ProcessSingle(Long64_t /*entry*/, Int_t /*i*/);
   virtual void        DoFill(Long64_t entry, Double_t w, const Double_t *v) = 0;
   virtual void        DefVar() = 0;

public:
   TProofDraw();
              ~TProofDraw() override;
   int         Version() const override { return 1; }
   void        Init(TTree *) override;
   void        Begin(TTree *) override;
   void        SlaveBegin(TTree *) override;
   Bool_t      Notify() override;
   Bool_t      Process(Long64_t /*entry*/) override;
   void        SlaveTerminate() override;
   void        Terminate() override;

   ClassDefOverride(TProofDraw,0)  //Tree drawing selector for PROOF
};


class TProofDrawHist : public TProofDraw {

private:
   void                DefVar1D();
   void                DefVar2D();
   void                DefVar3D();

protected:
   TH1                *fHistogram;

   virtual void        Begin1D(TTree *t);
   virtual void        Begin2D(TTree *t);
   virtual void        Begin3D(TTree *t);
   void        DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void        DefVar() override;

public:
   TProofDrawHist() : fHistogram(0) { }
   void        Begin(TTree *t) override;
   void        Init(TTree *) override;
   void        SlaveBegin(TTree *) override;
   void        Terminate() override;

   ClassDefOverride(TProofDrawHist,0)  //Tree drawing selector for PROOF
};


class TProofDrawEventList : public TProofDraw {

protected:
   TEventList*    fElist;          //  event list
   TList*         fEventLists;     //  a list of EventLists

   void   DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void   DefVar() override { }

public:
   TProofDrawEventList() : fElist(0), fEventLists(0) {}
   ~TProofDrawEventList() override {}

   void        Init(TTree *) override;
   void        SlaveBegin(TTree *) override;
   void        SlaveTerminate() override;
   void        Terminate() override;

   ClassDefOverride(TProofDrawEventList,0)  //Tree drawing selector for PROOF
};

class TProofDrawEntryList : public TProofDraw {
 protected:
   TEntryList *fElist;

   void DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void DefVar() override {}

 public:
   TProofDrawEntryList() : fElist(0) {}
   ~TProofDrawEntryList() override {}

   void Init(TTree *) override;
   void SlaveBegin(TTree *) override;
   void SlaveTerminate() override;
   void Terminate() override;

   ClassDefOverride(TProofDrawEntryList, 0)  //A Selectoor to fill a TEntryList from TTree::Draw
};


class TProofDrawProfile : public TProofDraw {

protected:
   TProfile           *fProfile;

   void        DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void        DefVar() override;

public:
   TProofDrawProfile() : fProfile(0) { }
   void        Init(TTree *) override;
   void        Begin(TTree *t) override;
   void        SlaveBegin(TTree *) override;
   void        Terminate() override;

   ClassDefOverride(TProofDrawProfile,0)  //Tree drawing selector for PROOF
};


class TProofDrawProfile2D : public TProofDraw {

protected:
   TProfile2D         *fProfile;

   void        DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void        DefVar() override;

public:
   TProofDrawProfile2D() : fProfile(0) { }
   void        Init(TTree *) override;
   void        Begin(TTree *t) override;
   void        SlaveBegin(TTree *) override;
   void        Terminate() override;

   ClassDefOverride(TProofDrawProfile2D,0)  //Tree drawing selector for PROOF
};


class TProofDrawGraph : public TProofDraw {

protected:
   TGraph             *fGraph;

   void        DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void        DefVar() override { }

public:
   TProofDrawGraph() : fGraph(0) { }
   void        Init(TTree *tree) override;
   void        SlaveBegin(TTree *) override;
   void        Terminate() override;

   ClassDefOverride(TProofDrawGraph,0)  //Tree drawing selector for PROOF
};


class TProofDrawPolyMarker3D : public TProofDraw {

protected:
   TPolyMarker3D      *fPolyMarker3D;

   void        DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void        DefVar() override { }

public:
   TProofDrawPolyMarker3D() : fPolyMarker3D(0) { }
   void        Init(TTree *tree) override;
   void        SlaveBegin(TTree *) override;
   void        Terminate() override;

   ClassDefOverride(TProofDrawPolyMarker3D,0)  //Tree drawing selector for PROOF
};

template <typename T>
class TProofVectorContainer : public TNamed {
   // Owns an std::vector<T>.
   // Implements Merge(TCollection*) which merges vectors holded
   // by all the TProofVectorContainers in the collection.
protected:
   std::vector<T> *fVector;   // vector

public:
   TProofVectorContainer(std::vector<T>* anVector) : fVector(anVector) { }
   TProofVectorContainer() : fVector(0) { }
   ~TProofVectorContainer() override { delete fVector; }

   std::vector<T> *GetVector() const { return fVector; }
   Long64_t        Merge(TCollection* list);

   ClassDefOverride(TProofVectorContainer,1) //Class describing a vector container
};

class TProofDrawListOfGraphs : public TProofDraw {

public:
   struct Point3D_t {
   public:
      Double_t fX, fY, fZ;
      Point3D_t(Double_t x, Double_t y, Double_t z) : fX(x), fY(y), fZ(z) { }
      Point3D_t() : fX(0), fY(0), fZ(0) { }
   };

protected:
   TProofVectorContainer<Point3D_t> *fPoints;
   void        DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void        DefVar() override { }

public:
   TProofDrawListOfGraphs() : fPoints(0) { }
   void        SlaveBegin(TTree *) override;
   void        Terminate() override;

   ClassDefOverride(TProofDrawListOfGraphs,0)  //Tree drawing selector for PROOF
};


class TProofDrawListOfPolyMarkers3D : public TProofDraw {

public:
   struct Point4D_t {
   public:
      Double_t fX, fY, fZ, fT;
      Point4D_t(Double_t x, Double_t y, Double_t z, Double_t t) : fX(x), fY(y), fZ(z), fT(t) { }
      Point4D_t() : fX(0), fY(0), fZ(0), fT(0) { }
   };

protected:
   TProofVectorContainer<Point4D_t> *fPoints;
   void        DoFill(Long64_t entry, Double_t w, const Double_t *v) override;
   void        DefVar() override { }

public:
   TProofDrawListOfPolyMarkers3D() : fPoints(0) { }
   void        SlaveBegin(TTree *) override;
   void        Terminate() override;

   ClassDefOverride(TProofDrawListOfPolyMarkers3D,0)  //Tree drawing selector for PROOF
};

#ifndef __CINT__
template <typename T>
Long64_t TProofVectorContainer<T>::Merge(TCollection* li)
{
   // Adds all vectors holded by all TProofVectorContainers in the collection
   // the vector holded by this TProofVectorContainer.
   // Returns the total number of poins in the result or -1 in case of an error.

   TIter next(li);

   std::back_insert_iterator<std::vector<T> > ii(*fVector);
   while (TObject* o = next()) {
      TProofVectorContainer<T> *vh = dynamic_cast<TProofVectorContainer<T>*> (o);
      if (!vh) {
         Error("Merge",
             "Cannot merge - an object which doesn't inherit from TProofVectorContainer<T> found in the list");
         return -1;
      }
      std::copy(vh->GetVector()->begin(), vh->GetVector()->end(), ii);
   }
   return fVector->size();
}
#endif

#endif
