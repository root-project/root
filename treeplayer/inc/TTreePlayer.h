// @(#)root/treeplayer:$Name:  $:$Id: TTreePlayer.h,v 1.11 2000/12/13 15:13:57 brun Exp $
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
#ifndef ROOT_TVirtualTreePlayer
#include "TVirtualTreePlayer.h"
#endif

class TTreeFormula;
class TH1;
class TSlave;
class TPacketGenerator;
class TSQLResult;
class TSelector;
class TPrincipal;

class TTreePlayer : public TVirtualTreePlayer {

protected:
    TTree         *fTree;           //  Pointer to current Tree
    TTreeFormula  *fVar1;           //  Pointer to first variable formula
    TTreeFormula  *fVar2;           //  Pointer to second variable formula
    TTreeFormula  *fVar3;           //  Pointer to third variable formula
    TTreeFormula  *fVar4;           //  Pointer to fourth variable formula
    TTreeFormula  *fSelect;         //  Pointer to selection formula
    TTreeFormula  *fMultiplicity;   //  Pointer to formula giving ndata per entry
    Int_t          fDraw;           //! Last entry loop number when object was drawn
    Int_t          fNfill;          //! Local for EntryLoop
    Int_t          fDimension;      //  Dimension of the current expression
    Int_t          fSelectedRows;   //  Number of selected entries
    Int_t          fPacketSize;     //  Number of entries in one packet for parallel root
    Int_t          fNbins[4];       //  Number of bins per dimension
    Double_t       fVmin[4];        //  Minima of varexp columns
    Double_t       fVmax[4];        //  Maxima of varexp columns
    Double_t      *fV1;             //[fSelectedRows]Local buffer for variable 1
    Double_t      *fV2;             //[fSelectedRows]Local buffer for variable 2
    Double_t      *fV3;             //[fSelectedRows]Local buffer for variable 3
    Double_t      *fW;              //[fSelectedRows]Local buffer for weights
    TPacketGenerator *fPacketGen;   //! Packet generator
    TH1           *fHistogram;      //! Pointer to histogram used for the projection

protected:
    const   char  *GetNameByIndex(TString &varexp, Int_t *index,Int_t colindex);
    virtual void    MakeIndex(TString &varexp, Int_t *index);
    void            TakeAction(Int_t nfill, Int_t &npoints, Int_t &action, TObject *obj, Option_t *option);
    void            TakeEstimate(Int_t nfill, Int_t &npoints, Int_t action, TObject *obj, Option_t *option);

public:
    TTreePlayer();
    virtual ~TTreePlayer();

    virtual void      ClearFormula();
    virtual void      CompileVariables(const char *varexp="", const char *selection="");
    virtual TTree    *CopyTree(const char *selection, Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      CreatePacketGenerator(Int_t nentries, Stat_t firstEntry);
    virtual Int_t     DrawSelect(const char *varexp, const char *selection, Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      EstimateLimits(Int_t estimate, Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      EntryLoop(Int_t &action, TObject *obj, Int_t nentries=1000000000, Int_t firstentry=0, Option_t *option="");

            void      FindGoodLimits(Int_t nbins, Int_t &newbins, Double_t &xmin, Double_t &xmax);
    virtual Int_t     Fit(const char *formula ,const char *varexp, const char *selection,Option_t *option ,Option_t *goption
                       ,Int_t nentries, Int_t firstentry);
    virtual Int_t     GetDimension() const {return fDimension;}
    TH1              *GetHistogram() const {return fHistogram;}
    TTreeFormula     *GetMultiplicity() const   {return fMultiplicity;}
    virtual void      GetNextPacket(TSlave *sl, Int_t &nentries, Stat_t &firstentry, Stat_t &processed);
    TPacketGenerator *GetPacketGenerator() const { return fPacketGen; }
    virtual Int_t     GetPacketSize() const {return fPacketSize;}
    TTreeFormula     *GetSelect() const    {return fSelect;}
    virtual Int_t     GetSelectedRows() const {return fSelectedRows;}
    TTreeFormula     *GetVar1() const {return fVar1;}
    TTreeFormula     *GetVar2() const {return fVar2;}
    TTreeFormula     *GetVar3() const {return fVar3;}
    TTreeFormula     *GetVar4() const {return fVar4;}
    virtual Double_t *GetV1() const   {return fV1;}
    virtual Double_t *GetV2() const   {return fV2;}
    virtual Double_t *GetV3() const   {return fV3;}
    virtual Double_t *GetW() const    {return fW;}
    virtual void      Loop(Option_t *option="",Int_t nentries=1000000000, Int_t firstentry=0);
    virtual Int_t     MakeClass(const char *classname=0, Option_t *option="");
    virtual Int_t     MakeCode(const char *filename=0);
    TPrincipal       *Principal(const char *varexp="", const char *selection="", Option_t *option="np"
                       ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual Int_t     Process(const char *filename,Option_t *option="", Int_t nentries=1000000000, Int_t firstentry=0);
    virtual Int_t     Process(TSelector *selector,Option_t *option="",  Int_t nentries=1000000000, Int_t firstentry=0);
    virtual Int_t     Scan(const char *varexp="", const char *selection="", Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual TSQLResult *Query(const char *varexp="", const char *selection="", Option_t *option=""
                         ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      SetEstimate(Int_t n);
    virtual void      SetPacketSize(Int_t size = 100);
    virtual void      SetTree(TTree *t) {fTree = t;}
    virtual void      StartViewer(Int_t ww, Int_t wh);
    virtual Int_t     UnbinnedFit(const char *formula ,const char *varexp, const char *selection,Option_t *option 
                       ,Int_t nentries, Int_t firstentry);

    ClassDef(TTreePlayer,1)  //manager class to play with TTrees
};

#endif
