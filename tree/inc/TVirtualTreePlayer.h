// @(#)root/tree:$Name:  $:$Id: TVirtualTreePlayer.h,v 1.10 2000/12/21 14:03:38 brun Exp $
// Author: Rene Brun   30/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualTreePlayer
#define ROOT_TVirtualTreePlayer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualTreePlayer                                                  //
//                                                                      //
// Abstract base class for Histogram pplayers                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TClass
#include "TClass.h"
#endif

class TTree;
class TH1;
class TTreeFormula;
class TSlave;
class TSQLResult;
class TSelector;
class TPrincipal;

class TVirtualTreePlayer : public TObject {


private:
   static TClass   *fgPlayer; //Pointer to Tree player

public:
    TVirtualTreePlayer() { }
    virtual ~TVirtualTreePlayer() { }
    virtual TTree         *CopyTree(const char *selection, Option_t *option=""
                            ,Int_t nentries=1000000000, Int_t firstentry=0) = 0;
    virtual Int_t          DrawSelect(const char *varexp, const char *selection, Option_t *option
                            ,Int_t nentries, Int_t firstentry) = 0;
    virtual Int_t          Fit(const char *formula ,const char *varexp, const char *selection,Option_t *option ,Option_t *goption
                            ,Int_t nentries, Int_t firstentry) = 0;
    virtual Int_t          GetDimension() const = 0;
    virtual TH1           *GetHistogram() const = 0;
    virtual void           GetNextPacket(TSlave *sl, Int_t &nentries, Stat_t &firstentry, Stat_t &processed) = 0;
    virtual Int_t          GetSelectedRows() const = 0;
    virtual TTreeFormula  *GetSelect() const = 0;
    virtual TTreeFormula  *GetVar1() const = 0;
    virtual TTreeFormula  *GetVar2() const = 0;
    virtual TTreeFormula  *GetVar3() const = 0;
    virtual TTreeFormula  *GetVar4() const = 0;
    virtual Double_t      *GetV1() const = 0;
    virtual Double_t      *GetV2() const = 0;
    virtual Double_t      *GetV3() const = 0;
    virtual Double_t      *GetW() const = 0;
    virtual void           Loop(Option_t *option,Int_t nentries, Int_t firstentry) = 0;
    virtual Int_t          MakeClass(const char *classname, const char *option) = 0;
    virtual Int_t          MakeCode(const char *filename) = 0;
    virtual TPrincipal    *Principal(const char *varexp="", const char *selection="", Option_t *option="np"
                           ,Int_t nentries=1000000000, Int_t firstentry=0) = 0;
    virtual Int_t          Process(const char *filename,Option_t *option="", Int_t nentries=1000000000, Int_t firstentry=0) = 0;
    virtual Int_t          Process(TSelector *selector,Option_t *option="",  Int_t nentries=1000000000, Int_t firstentry=0) = 0;
    virtual Int_t          Scan(const char *varexp, const char *selection, Option_t *option
                            ,Int_t nentries, Int_t firstentry) = 0;
    virtual TSQLResult    *Query(const char *varexp, const char *selection, Option_t *option
                            ,Int_t nentries, Int_t firstentry) = 0;
    virtual void           SetEstimate(Int_t n) = 0;
    virtual void           SetPacketSize(Int_t) = 0;
    virtual void           SetTree(TTree *t) = 0;
    virtual void           StartViewer(Int_t ww, Int_t wh) = 0;
    virtual Int_t          UnbinnedFit(const char *formula ,const char *varexp, const char *selection,Option_t *option
                            ,Int_t nentries, Int_t firstentry) = 0;
    virtual void           UpdateFormulaLeaves() = 0;

   static  TVirtualTreePlayer *TreePlayer(TTree *obj);
   static void        SetPlayer(const char *player);

    ClassDef(TVirtualTreePlayer,0)  //Abstract interface for Tree players
};

#endif
