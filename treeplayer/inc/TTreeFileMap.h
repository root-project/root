// @(#)root/treeplayer:$Name:  $:$Id: TTreeFileMap.h,v 1.22 2003/01/10 14:51:50 brun Exp $
// Author: Rene Brun   15/01/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeFileMap
#define ROOT_TTreeFileMap


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeFileMap                                                         //
//                                                                      //
// Draw a 2-d map -f the branches of a Tree in its file                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TTree;
class TH1;

class TTreeFileMap : public TObject {

protected:
    TTree         *fTree;           //!  Pointer to current Tree
    TH1           *fFrame;          //histogram used to draw the map frame
    TString        fBranches;       //list of branches 
    TString        fOption;         //drawing options
    Int_t          fXsize;          //size in bytes of X axis
    Int_t          fYsize;          //size in K/Mbytes of Y axis
           
public:
    TTreeFileMap();
    TTreeFileMap(TTree *tree, const char *branches, Option_t *option);
    virtual ~TTreeFileMap();

    virtual Int_t DistancetoPrimitive(Int_t px, Int_t py);
    virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual char *GetObjectInfo(Int_t px, Int_t py) const;
    virtual void  Paint(Option_t *option);
    virtual void  SavePrimitive(ofstream &out, Option_t *option);
    virtual void  ShowEntry(); // *MENU*
    
    ClassDef(TTreeFileMap,1)  //Draw a 2-d map -f the branches of a Tree in its file
};

#endif
