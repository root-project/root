// @(#)root/treeplayer:$Name:  $:$Id: TSelectorDraw.h,v 1.10 2005/11/11 23:21:43 pcanal Exp $
// Author: Rene Brun   08/01/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelectorDraw
#define ROOT_TSelectorDraw


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelectorDraw                                                        //
//                                                                      //
// A specialized TSelector for TTree::Draw.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSelector
#include "TSelector.h"
#endif

class TTreeFormula;
class TTreeFormulaManager;
class TH1;

class TSelectorDraw : public TSelector {

protected:
   enum { kWarn = BIT(12) };

   TTree         *fTree;           //  Pointer to current Tree
   TTreeFormula  *fVar1;           //  Pointer to first variable formula
   TTreeFormula  *fVar2;           //  Pointer to second variable formula
   TTreeFormula  *fVar3;           //  Pointer to third variable formula
   TTreeFormula  *fVar4;           //  Pointer to fourth variable formula
   TTreeFormula  *fSelect;         //  Pointer to selection formula
   TTreeFormulaManager *fManager;  //  Pointer to the formula manager
   TObject       *fObject;         //! Pointer to object being filled (histogram, event list)
   TObject       *fTreeElist;      //  pointer to Tree Event list
   TH1           *fOldHistogram;   //! Pointer to previously used histogram
   Int_t          fAction;         //! Action type
   Long64_t       fDraw;           //! Last entry loop number when object was drawn
   Int_t          fNfill;          //! Total number of histogram fills
   Int_t          fMultiplicity;   //  Indicator of the variability of the size of entries
   Int_t          fDimension;      //  Dimension of the current expression
   Long64_t       fSelectedRows;   //  Number of selected entries
   Long64_t       fOldEstimate;    //  value of Tree fEstimate when selector is called
   Int_t          fForceRead;      //  Forec Read flag
   Int_t          fNbins[4];       //  Number of bins per dimension
   Double_t       fVmin[4];        //  Minima of varexp columns
   Double_t       fVmax[4];        //  Maxima of varexp columns
   Double_t       fWeight;         //  Tree weight (see TTree::SetWeight)
   Double_t      *fV1;             //![fSelectedRows]Local buffer for variable 1
   Double_t      *fV2;             //![fSelectedRows]Local buffer for variable 2
   Double_t      *fV3;             //![fSelectedRows]Local buffer for variable 3
   Double_t      *fV4;             //![fSelectedRows]Local buffer for variable 4
   Double_t      *fW;              //![fSelectedRows]Local buffer for weights
   Bool_t         fVar1Multiple;   //  true if var1 has a variable index
   Bool_t         fVar2Multiple;   //  true if var2 has a variable index
   Bool_t         fVar3Multiple;   //  true if var3 has a variable index
   Bool_t         fVar4Multiple;   //  true if var4 has a variable index
   Bool_t         fSelectMultiple; //  true if selection has a variable index
   Bool_t         fCleanElist;     //  true if original Tree elist must be saved
   Bool_t         fObjEval;        //  true if fVar1 returns an object (or pointer to).

protected:
   TSelectorDraw(const TSelectorDraw&);
   TSelectorDraw& operator=(const TSelectorDraw&);

   virtual void      ClearFormula();
   virtual Bool_t    CompileVariables(const char *varexp="", const char *selection="");

public:
   TSelectorDraw();
   virtual ~TSelectorDraw();

   virtual void      Begin(TTree *tree);
   virtual Int_t     GetAction() const {return fAction;}
   virtual Bool_t    GetCleanElist() const {return fCleanElist;}
   virtual Int_t     GetDimension() const {return fDimension;}
   virtual Int_t     GetDrawFlag() const {return fDraw;}
   TObject          *GetObject() const {return fObject;}
   Int_t             GetMultiplicity() const   {return fMultiplicity;}
   const char       *GetNameByIndex(TString &varexp, Int_t *index,Int_t colindex);
   virtual Int_t     GetNfill() const {return fNfill;}
   TH1              *GetOldHistogram() const {return fOldHistogram;}
   TTreeFormula     *GetSelect() const    {return fSelect;}
   virtual Long64_t  GetSelectedRows() const {return fSelectedRows;}
   TTreeFormula     *GetVar1() const {return fVar1;}
   TTreeFormula     *GetVar2() const {return fVar2;}
   TTreeFormula     *GetVar3() const {return fVar3;}
   TTreeFormula     *GetVar4() const {return fVar4;}
   virtual Double_t *GetV1() const   {return fV1;}
   virtual Double_t *GetV2() const   {return fV2;}
   virtual Double_t *GetV3() const   {return fV3;}
   virtual Double_t *GetV4() const   {return fV4;}
   virtual Double_t *GetW() const    {return fW;}
   virtual void      MakeIndex(TString &varexp, Int_t *index);
   virtual Bool_t    Notify();
   virtual Bool_t    Process(Long64_t /*entry*/) { return kFALSE; }
   virtual void      ProcessFill(Long64_t entry);
   virtual void      ProcessFillMultiple(Long64_t entry);
   virtual void      ProcessFillObject(Long64_t entry);
   virtual void      SetEstimate(Long64_t n);
   virtual void      TakeAction();
   virtual void      TakeEstimate();
   virtual void      Terminate();

   ClassDef(TSelectorDraw,1);  //A specialized TSelector for TTree::Draw
};

#endif
