// @(#)root/treeplayer:$Name:  $:$Id: TTreePlayer.h,v 1.24 2003/01/17 17:48:56 brun Exp $
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreePlayer
#define ROOT_TTreePlayer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreePlayer                                                          //
//                                                                      //
// A TTree object is a list of TBranch.                                 //
//   To Create a TTree object one must:                                 //
//    - Create the TTree header via the TTree constructor               //
//    - Call the TBranch constructor for every branch.                  //
//                                                                      //
//   To Fill this object, use member function Fill with no parameters.  //
//     The Fill function loops on all defined TBranch.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TSelectorDraw
#include "TSelectorDraw.h"
#endif
//#ifndef ROOT_TTreeFormula
//#include "TTreeFormula.h"
//#endif
#ifndef ROOT_TVirtualTreePlayer
#include "TVirtualTreePlayer.h"
#endif

class TTreePlayer : public TVirtualTreePlayer {

protected:
    TTree         *fTree;           //!  Pointer to current Tree
    Bool_t         fScanRedirect;   //  Switch to redirect TTree::Scan output to a file
    const char    *fScanFileName;   //  Name of the file where Scan is redirected
    Int_t          fDimension;      //  Dimension of the current expression
    Int_t          fSelectedRows;   //  Number of selected entries
    TH1           *fHistogram;      //! Pointer to histogram used for the projection
    TSelectorDraw *fSelector;       //! Pointer to current selector
    TList         *fInput;          //! input list to the selector
    TList         *fFormulaList;    //! Pointer to a list of coordinated list TTreeFormula (used by Scan and Query)
    
protected:
    const   char  *GetNameByIndex(TString &varexp, Int_t *index,Int_t colindex);
    void           TakeAction(Int_t nfill, Int_t &npoints, Int_t &action, TObject *obj, Option_t *option);
    void           TakeEstimate(Int_t nfill, Int_t &npoints, Int_t action, TObject *obj, Option_t *option);

public:
    TTreePlayer();
    virtual ~TTreePlayer();

    virtual TTree    *CopyTree(const char *selection, Option_t *option
                       ,Int_t nentries, Int_t firstentry);
    virtual Int_t     DrawSelect(const char *varexp, const char *selection, Option_t *option
                       ,Int_t nentries, Int_t firstentry);
    virtual Int_t     Fit(const char *formula ,const char *varexp, const char *selection,Option_t *option ,
                        Option_t *goption ,Int_t nentries, Int_t firstentry);
    virtual Int_t     GetDimension() const {return fDimension;}
    TH1              *GetHistogram() const {return fHistogram;}
    virtual Int_t     GetNfill() const {return fSelector->GetNfill();}
    const char       *GetScanFileName() const {return fScanFileName;}
    TTreeFormula     *GetSelect() const    {return fSelector->GetSelect();}
    virtual Int_t     GetSelectedRows() const {return fSelectedRows;}
    TSelector        *GetSelector() const {return fSelector;}
    TTreeFormula     *GetVar1() const {return fSelector->GetVar1();}
    TTreeFormula     *GetVar2() const {return fSelector->GetVar2();}
    TTreeFormula     *GetVar3() const {return fSelector->GetVar3();}
    TTreeFormula     *GetVar4() const {return fSelector->GetVar4();}
    virtual Double_t *GetV1() const   {return fSelector->GetV1();}
    virtual Double_t *GetV2() const   {return fSelector->GetV2();}
    virtual Double_t *GetV3() const   {return fSelector->GetV3();}
    virtual Double_t *GetW() const    {return fSelector->GetW();}
    virtual Int_t     MakeClass(const char *classname, Option_t *option);
    virtual Int_t     MakeCode(const char *filename);
    TPrincipal       *Principal(const char *varexp, const char *selection, Option_t *option
                       ,Int_t nentries, Int_t firstentry);
    virtual Int_t     Process(const char *filename,Option_t *option, Int_t nentries, Int_t firstentry);
    virtual Int_t     Process(TSelector *selector,Option_t *option,  Int_t nentries, Int_t firstentry);
    virtual Int_t     Scan(const char *varexp, const char *selection, Option_t *option
                       ,Int_t nentries, Int_t firstentry);
    Bool_t            ScanRedirected() {return fScanRedirect;}
    virtual TSQLResult *Query(const char *varexp, const char *selection, Option_t *option
                         ,Int_t nentries, Int_t firstentry);
    virtual void      SetEstimate(Int_t n);
    void              SetScanRedirect(Bool_t on=kFALSE) {fScanRedirect = on;}
    void              SetScanFileName(const char *name) {fScanFileName=name;}
    virtual void      SetTree(TTree *t) {fTree = t;}
    virtual void      StartViewer(Int_t ww, Int_t wh);
    virtual Int_t     UnbinnedFit(const char *formula ,const char *varexp, const char *selection,Option_t *option
                       ,Int_t nentries, Int_t firstentry);
    virtual void      UpdateFormulaLeaves();

    ClassDef(TTreePlayer,2)  //Manager class to play with TTrees
};

#endif
