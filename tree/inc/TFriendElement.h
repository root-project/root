// @(#)root/tree:$Name:  $:$Id: TFriendElement.h,v 1.5 2001/01/23 21:09:08 brun Exp $
// Author: Rene Brun   07/04/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TFriendElement
#define ROOT_TFriendElement


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFriendElement                                                       //
//                                                                      //
// A TFriendElement TF describes a TTree object TF in a file.           //
// When a TFriendElement TF is added to the the list of friends of an   //
// existing TTree T, any variable from TF can be referenced in a query  //
// to T.                                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TFile;
class TTree;

class TFriendElement : public TNamed {

protected:
    TTree        *fParentTree;      //!pointer  to the parent TTree
    TTree        *fTree;            //!pointer  to the TTree described by this element
    TFile        *fFile;            //!pointer to the file containing the friend Tree

public:
        TFriendElement();
        TFriendElement(TTree *tree, const char *treename, const char *filename);
        virtual ~TFriendElement();
        virtual TTree   *Connect();
        virtual TTree   *DisConnect();
        virtual TFile   *GetFile();
        virtual TTree   *GetParentTree() const {return fParentTree;}
        virtual TTree   *GetTree();
        virtual void     ls(Option_t *option="") const;

        ClassDef(TFriendElement,1)  //A friend element of another TTree
};

#endif

