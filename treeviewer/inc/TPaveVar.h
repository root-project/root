// @(#)root/treeviewer:$Name:  $:$Id: TPaveVar.h,v 1.2 2000/06/13 13:59:21 brun Exp $
// Author: Rene Brun   08/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TPaveVar
#define ROOT_TPaveVar


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPaveVar                                                             //
//                                                                      //
// A TPaveLabel specialized for TTree variables and cuts                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPaveLabel
#include "TPaveLabel.h"
#endif
#ifndef ROOT_TTreeViewer
#include "TTreeViewerOld.h"
#endif


class TPaveVar : public TPaveLabel{

protected:
        TTreeViewer   *fViewer;       //Pointer to the TTreeViewer referencing this object
        virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);

public:
        // TPaveVar status bits
        enum { kBranchObject = BIT(15) };

        TPaveVar();
        TPaveVar(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, const char *label, TTreeViewer *viewer);
        TPaveVar(const TPaveVar &PaveVar);
        virtual      ~TPaveVar();
                void  Copy(TObject &PaveVar);
        TTreeViewer  *GetViewer() {return fViewer;}
        virtual void  SavePrimitive(ofstream &out, Option_t *option);
        virtual void  Merge(Option_t *option="AND");  // *MENU*
        virtual void  SetViewer(TTreeViewer *viewer) {fViewer = viewer;}

        ClassDef(TPaveVar,1)  //A TPaveLabel specialized for TTree variables and cuts
};

#endif

