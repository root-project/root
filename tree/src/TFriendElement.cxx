// @(#)root/tree:$Name:  $:$Id: TFriendElement.cxx,v 1.3 2001/04/27 08:03:06 brun Exp $
// Author: Rene Brun   11/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFriendElement                                                       //
//                                                                      //
// A TFriendElement TF describes a TTree object TF in a file.           //
// When a TFriendElement TF is added to the the list of friends of an   //
// existing TTree T, any variable from TF can be referenced in a query  //
// to T.                                                                //
//                                                                      //
// To add a TFriendElement to an existing TTree T, do:                  //
//       T.AddFriend("friendTreename","friendTreeFile");               //
//                                                                      //
//  see TTree::AddFriend for more information                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTree.h"
#include "TFriendElement.h"
#include "TFile.h"

R__EXTERN TTree *gTree;

ClassImp(TFriendElement)

//______________________________________________________________________________
TFriendElement::TFriendElement(): TNamed()
{
//*-*-*-*-*-*Default constructor for a friend element*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =======================================

   fFile       = 0;
   fTree       = 0;
   fParentTree = gTree;
}

//______________________________________________________________________________
TFriendElement::TFriendElement(TTree *tree, const char *treename, const char *filename)
    :TNamed(treename,filename)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a friend element*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ======================
//
// if treename is of the form "a=b", an alias called "a" is created for treename = "b"
// by default the alias name is the name of the tree.   

   fFile       = 0;
   fTree       = 0;
   fParentTree = tree;
   fTreeName   = treename;
   if (strchr(treename,'=')) {
      char *temp = Compress(treename);
      char *equal = strchr(temp,'=');
      *equal=0;
      fTreeName = equal+1;
      SetName(temp);
      delete [] temp;
   }
   
   Connect();
}

//______________________________________________________________________________
TFriendElement::~TFriendElement()
{

   DisConnect(); 
}

//_______________________________________________________________________
TTree *TFriendElement::Connect()
{
// Connect file and Tree

   GetFile();
   return GetTree();
}

//_______________________________________________________________________
TTree *TFriendElement::DisConnect()
{
// DisConnect file and Tree

   if (fTree) delete fTree;
   TDirectory *dir = fParentTree->GetDirectory();
   if (dir && fFile != dir->GetFile()) delete fFile;
   fFile = 0;
   fTree = 0;
   return 0;
}

//_______________________________________________________________________
TFile *TFriendElement::GetFile()
{
// return pointer to TFile containing this friend Tree

   if (fFile) return fFile;
   if (strlen(GetTitle())) fFile = new TFile(GetTitle());
   else {
      TDirectory *dir = fParentTree->GetDirectory();
      if (dir) fFile = dir->GetFile();
   }
   if (fFile && fFile->IsZombie()) {
      delete fFile;
      fFile = 0;
   }
   return fFile;
}

//_______________________________________________________________________
TTree *TFriendElement::GetTree()
{
// return pointer to friend Tree

   if (fTree) return fTree;
   if (!GetFile()) return 0;
   fTree = (TTree*)fFile->Get(GetTreeName());
   TDirectory *dir = fParentTree->GetDirectory();
   if (dir && dir != gDirectory) dir->cd();
   return fTree;
}

//_______________________________________________________________________
void TFriendElement::ls(Option_t *option) const
{
// List this friend element
//

   printf(" friend Tree: %s in file: %s\n",GetName(),GetTitle());
}
