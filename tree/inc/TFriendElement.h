// @(#)root/tree:$Name:  $:$Id: TFriendElement.h,v 1.3 2001/09/22 10:06:55 rdm Exp $
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
    TTree        *fParentTree;  //!pointer to the parent TTree
    TTree        *fTree;        //!pointer to the TTree described by this element
    TFile        *fFile;        //!pointer to the file containing the friend TTree
    Bool_t        fOwnTree;     //!true if tree is managed by this class
    TString       fTreeName;    // name of the friend TTree
    Bool_t        fOwnFile;     // true if file is managed by this class

public:
   TFriendElement();
   TFriendElement(TTree *tree, const char *treename, const char *filename);
   TFriendElement(TTree *tree, const char *treename, TFile *file);
   TFriendElement(TTree *tree, TTree* friendtree, const char *alias);
   virtual ~TFriendElement();
   virtual TTree      *Connect();
   virtual TTree      *DisConnect();
   virtual TFile      *GetFile();
   virtual TTree      *GetParentTree() const {return fParentTree;}
   virtual TTree      *GetTree();
   virtual const char *GetTreeName() const {return fTreeName.Data();}
   virtual void        ls(Option_t *option="") const;

   ClassDef(TFriendElement,2)  //A friend element of another TTree
};

#endif

