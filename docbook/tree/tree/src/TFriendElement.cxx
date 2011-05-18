// @(#)root/tree:$Id$
// Author: Rene Brun   07/04/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
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
//       T.AddFriend("friendTreename","friendTreeFile");                //
//                                                                      //
//  See TTree::AddFriend for more information.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTree.h"
#include "TFriendElement.h"
#include "TFile.h"
#include "TROOT.h"

R__EXTERN TTree *gTree;

ClassImp(TFriendElement)

//______________________________________________________________________________
TFriendElement::TFriendElement() : TNamed()
{
//*-*-*-*-*-*Default constructor for a friend element*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =======================================

   fFile       = 0;
   fTree       = 0;
   fOwnFile    = kFALSE;
   fParentTree = gTree;
}

//______________________________________________________________________________
TFriendElement::TFriendElement(TTree *tree, const char *treename, const char *filename)
    :TNamed(treename,filename)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a friend element*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ======================
//
// If treename is of the form "a=b", an alias called "a" is created for
// treename = "b" by default the alias name is the name of the tree.

   fFile       = 0;
   fTree       = 0;
   fOwnFile    = kTRUE;
   fParentTree = tree;
   fTreeName   = treename;
   if (strchr(treename,'=')) {
      char *temp = Compress(treename);
      char *equal = strchr(temp,'=');
      if (!equal) return;;
      *equal=0;
      fTreeName = equal+1;
      SetName(temp);
      delete [] temp;
   }

   Connect();
}

//______________________________________________________________________________
TFriendElement::TFriendElement(TTree *tree, const char *treename, TFile *file)
    :TNamed(treename,file?file->GetName():"")
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a friend element*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ======================
//
// If treename is of the form "a=b", an alias called "a" is created for
// treename = "b" by default the alias name is the name of the tree.
// The passed TFile is managed by the user (i.e. user must delete the TFile).

   fFile       = file;
   fTree       = 0;
   fOwnFile    = kFALSE;
   fParentTree = tree;
   fTreeName   = treename;
   if (fParentTree && fParentTree->GetDirectory() 
       && fParentTree->GetDirectory()->GetFile() == fFile) {
      // The friend and the TTree are in the same file, let's not record
      // the filename.
      SetTitle("");
   }
   if (strchr(treename,'=')) {
      char *temp = Compress(treename);
      char *equal = strchr(temp,'=');
      if (!equal) return;;
      *equal=0;
      fTreeName = equal+1;
      SetName(temp);
      delete [] temp;
   }

   Connect();
}

//______________________________________________________________________________
TFriendElement::TFriendElement(TTree *tree, TTree* friendtree, const char *alias)
   : TNamed(friendtree?friendtree->GetName():"",
            friendtree
            ? (   friendtree->GetDirectory()
                  ? (    friendtree->GetDirectory()->GetFile()
                         ? friendtree->GetDirectory()->GetFile()->GetName()
                         :  "")
                  : "")
            :  "")
{
   // Create a friend element.

   fTree       = friendtree;
   fTreeName   = "";
   fFile       = 0;
   fOwnFile    = kFALSE;
   fParentTree = tree;
   if (fTree) {
      fTreeName   = fTree->GetName();
      if (fTree->GetDirectory()) fFile = fTree->GetDirectory()->GetFile();
      if (fParentTree && fParentTree->GetDirectory() 
          && fParentTree->GetDirectory()->GetFile() == fFile) {
         // The friend and the TTree are in the same file, let's not record
         // the filename.
         SetTitle("");
      }
   }
   if (alias && strlen(alias)) {
      char *temp = Compress(alias);
      SetName(temp);
      delete [] temp;
   }

   // No need to Connect.
}

//______________________________________________________________________________
TFriendElement::TFriendElement(const TFriendElement& tfe) : 
   TNamed(tfe),
   fParentTree(tfe.fParentTree),
   fTree(tfe.fTree),
   fFile(tfe.fFile),
   fTreeName(tfe.fTreeName),
   fOwnFile(tfe.fOwnFile)
{ 
   // Copy constructor
}

//______________________________________________________________________________
TFriendElement& TFriendElement::operator=(const TFriendElement& tfe) 
{
   // Equal operator
   if(this!=&tfe) {
      TNamed::operator=(tfe);
      fParentTree=tfe.fParentTree;
      fTree=tfe.fTree;
      fFile=tfe.fFile;
      fTreeName=tfe.fTreeName;
      fOwnFile=tfe.fOwnFile;
   } return *this;
}

//______________________________________________________________________________
TFriendElement::~TFriendElement()
{
   // Destructor.  Disconnect from the owning tree if needed.

   DisConnect();
}

//_______________________________________________________________________
TTree *TFriendElement::Connect()
{
   // Connect file and return TTree.

   GetFile();
   return GetTree();
}

//_______________________________________________________________________
TTree *TFriendElement::DisConnect()
{
   // DisConnect file and TTree.

   if (fOwnFile) delete fFile;
   fFile = 0;
   fTree = 0;
   return 0;
}

//_______________________________________________________________________
TFile *TFriendElement::GetFile()
{
   // Return pointer to TFile containing this friend TTree.

   if (fFile || IsZombie()) return fFile;


   if (strlen(GetTitle())) {
      TDirectory::TContext ctxt(gDirectory, 0);
      fFile = TFile::Open(GetTitle());
      fOwnFile = kTRUE;
   } else {
      TDirectory *dir = fParentTree->GetDirectory();
      if (dir) {
         fFile = dir->GetFile();
         fOwnFile = kFALSE;
      }
   }
   if (fFile && fFile->IsZombie()) {
      MakeZombie();
      delete fFile;
      fFile = 0;
   }
   return fFile;
}

//_______________________________________________________________________
TTree *TFriendElement::GetTree()
{
   // Return pointer to friend TTree.

   if (fTree) return fTree;

   if (GetFile()) {
      fFile->GetObject(GetTreeName(),fTree);
      if (fTree) return fTree;
   }

   // This could be a memory tree or chain
   fTree = dynamic_cast<TTree*>( gROOT->FindObject(GetTreeName()) );

   return fTree;
}

//_______________________________________________________________________
void TFriendElement::ls(Option_t *) const
{
   // List this friend element.

   printf(" Friend Tree: %s in file: %s\n",GetName(),GetTitle());
}
