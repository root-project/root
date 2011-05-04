// @(#)root/tree:$Id$
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBranch.h"

#include "TBasket.h"
#include "TBranchBrowsable.h"
#include "TBrowser.h"
#include "TClass.h"
#include "TBufferFile.h"
#include "TClonesArray.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TLeafB.h"
#include "TLeafC.h"
#include "TLeafD.h"
#include "TLeafF.h"
#include "TLeafI.h"
#include "TLeafL.h"
#include "TLeafO.h"
#include "TLeafObject.h"
#include "TLeafS.h"
#include "TMessage.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TMath.h"
#include "TTree.h"
#include "TTreeCache.h"
#include "TTreeCacheUnzip.h"
#include "TVirtualPad.h"

#include <cstddef>
#include <string.h>
#include <stdio.h>

R__EXTERN TTree* gTree;

Int_t TBranch::fgCount = 0;


#if (__GNUC__ >= 3) || defined(__INTEL_COMPILER)
#if !defined(R__unlikely)
  #define R__unlikely(expr) __builtin_expect(!!(expr), 0)
#endif
#if !defined(R__likely)
  #define R__likely(expr) __builtin_expect(!!(expr), 1)
#endif
#else
  #define R__unlikely(expr) expr
  #define R__likely(expr) expr
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TTree is a list of TBranches                                       //
//                                                                      //
// A TBranch supports:                                                  //
//   - The list of TLeaf describing this branch.                        //
//   - The list of TBasket (branch buffers).                            //
//                                                                      //
//       See TBranch structure in TTree.                                //
//                                                                      //
// See also specialized branches:                                       //
//     TBranchObject in case the branch is one object                   //
//     TBranchClones in case the branch is an array of clone objects    //
//////////////////////////////////////////////////////////////////////////

ClassImp(TBranch)

//______________________________________________________________________________
TBranch::TBranch()
: TNamed()
, TAttFill(0, 1001)
, fCompress(0)
, fBasketSize(32000)
, fEntryOffsetLen(1000)
, fWriteBasket(0)
, fEntryNumber(0)
, fOffset(0)
, fMaxBaskets(10)
, fNBaskets(0)
, fSplitLevel(0)
, fNleaves(0)
, fReadBasket(0)
, fReadEntry(-1)
, fFirstBasketEntry(-1)
, fNextBasketEntry(-1)
, fCurrentBasket(0)
, fEntries(0)
, fFirstEntry(0)
, fTotBytes(0)
, fZipBytes(0)
, fBranches()
, fLeaves()
, fBaskets(fMaxBaskets)
, fBasketBytes(0)
, fBasketEntry(0)
, fBasketSeek(0)
, fTree(0)
, fMother(0)
, fParent(0)
, fAddress(0)
, fDirectory(0)
, fFileName("")
, fEntryBuffer(0)
, fBrowsables(0)
, fSkipZip(kFALSE)
, fReadLeaves(&TBranch::ReadLeavesImpl)
{
   // Default constructor.  Used for I/O by default.

   SetBit(TBranch::kDoNotUseBufferMap);
}

//______________________________________________________________________________
TBranch::TBranch(TTree *tree, const char* name, void* address, const char* leaflist, Int_t basketsize, Int_t compress)
: TNamed(name, leaflist)
, TAttFill(0, 1001)
, fCompress(compress)
, fBasketSize((basketsize < 100) ? 100 : basketsize)
, fEntryOffsetLen(0)
, fWriteBasket(0)
, fEntryNumber(0)
, fOffset(0)
, fMaxBaskets(10)
, fNBaskets(0)
, fSplitLevel(0)
, fNleaves(0)
, fReadBasket(0)
, fReadEntry(-1)
, fFirstBasketEntry(-1)
, fNextBasketEntry(-1)
, fCurrentBasket(0)
, fEntries(0)
, fFirstEntry(0)
, fTotBytes(0)
, fZipBytes(0)
, fBranches()
, fLeaves()
, fBaskets(fMaxBaskets)
, fBasketBytes(0)
, fBasketEntry(0)
, fBasketSeek(0)
, fTree(tree)
, fMother(0)
, fParent(0)
, fAddress((char*) address)
, fDirectory(fTree->GetDirectory())
, fFileName("")
, fEntryBuffer(0)
, fBrowsables(0)
, fSkipZip(kFALSE)
, fReadLeaves(&TBranch::ReadLeavesImpl)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*Create a Branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                =====================
   //
   //       * address is the address of the first item of a structure.
   //       * leaflist is the concatenation of all the variable names and types
   //         separated by a colon character :
   //         The variable name and the variable type are separated by a slash (/).
   //         The variable type may be 0,1 or 2 characters. If no type is given,
   //         the type of the variable is assumed to be the same as the previous
   //         variable. If the first variable does not have a type, it is assumed
   //         of type F by default. The list of currently supported types is given below:
   //            - C : a character string terminated by the 0 character
   //            - B : an 8 bit signed integer (Char_t)
   //            - b : an 8 bit unsigned integer (UChar_t)
   //            - S : a 16 bit signed integer (Short_t)
   //            - s : a 16 bit unsigned integer (UShort_t)
   //            - I : a 32 bit signed integer (Int_t)
   //            - i : a 32 bit unsigned integer (UInt_t)
   //            - F : a 32 bit floating point (Float_t)
   //            - D : a 64 bit floating point (Double_t)
   //            - L : a 64 bit signed integer (Long64_t)
   //            - l : a 64 bit unsigned integer (ULong64_t)
   //
   //         By default, a variable will be copied to the buffer with the number of
   //         bytes specified in the type descriptor character. However, if the type
   //         consists of 2 characters, the second character is an integer that
   //         specifies the number of bytes to be used when copying the variable
   //         to the output buffer. Example:
   //             X         ; variable X, type Float_t
   //             Y/I       : variable Y, type Int_t
   //             Y/I2      ; variable Y, type Int_t converted to a 16 bits integer
   //
   //         Arrays of values are supported with the following syntax:
   //         If leaf name has the form var[nelem], where nelem is alphanumeric, then
   //            if nelem is a leaf name, it is used as the variable size of the array, 
   //            otherwise return 0.
   //            The leaf refered to by neleme **MUST** be an int (/I), 
   //         If leaf name has the form var[nelem], where nelem is a digit, then
   //            it is used as the fixed size of the array.
   //         If leaf name has the form of a multi dimenantion array (eg var[nelem][nelem2])
   //            where nelem and nelem2 are digits) then
   //            it is used as a 2 dimensional array of fixed size.
   //         Any of other form is not supported.
   //
   //    Note that the TTree will assume that all the item are contiguous in memory.
   //    On some platform, this is not always true of the member of a struct or a class,
   //    due to padding and alignment.  Sorting your data member in order of decreasing
   //    sizeof usually leads to their being contiguous in memory.
   //
   //       * bufsize is the buffer size in bytes for this branch
   //         The default value is 32000 bytes and should be ok for most cases.
   //         You can specify a larger value (eg 256000) if your Tree is not split
   //         and each entry is large (Megabytes)
   //         A small value for bufsize is optimum if you intend to access
   //         the entries in the Tree randomly and your Tree is in split mode.
   //
   //   See an example of a Branch definition in the TTree constructor.
   //
   //   Note that in case the data type is an object, this branch can contain
   //   only this object.
   //
   //    Note that this function is invoked by TTree::Branch
   //
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Init(name,leaflist,compress);
}

//______________________________________________________________________________
TBranch::TBranch(TBranch *parent, const char* name, void* address, const char* leaflist, Int_t basketsize, Int_t compress)
: TNamed(name, leaflist)
, TAttFill(0, 1001)
, fCompress(compress)
, fBasketSize((basketsize < 100) ? 100 : basketsize)
, fEntryOffsetLen(0)
, fWriteBasket(0)
, fEntryNumber(0)
, fOffset(0)
, fMaxBaskets(10)
, fNBaskets(0)
, fSplitLevel(0)
, fNleaves(0)
, fReadBasket(0)
, fReadEntry(-1)
, fFirstBasketEntry(-1)
, fNextBasketEntry(-1)
, fCurrentBasket(0)
, fEntries(0)
, fFirstEntry(0)
, fTotBytes(0)
, fZipBytes(0)
, fBranches()
, fLeaves()
, fBaskets(fMaxBaskets)
, fBasketBytes(0)
, fBasketEntry(0)
, fBasketSeek(0)
, fTree(parent ? parent->GetTree() : 0)
, fMother(parent ? parent->GetMother() : 0)
, fParent(parent)
, fAddress((char*) address)
, fDirectory(fTree ? fTree->GetDirectory() : 0)
, fFileName("")
, fEntryBuffer(0)
, fBrowsables(0)
, fSkipZip(kFALSE)
, fReadLeaves(&TBranch::ReadLeavesImpl)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*Create a Branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                =====================
   //
   //       * address is the address of the first item of a structure
   //         or the address of a pointer to an object (see example).
   //       * leaflist is the concatenation of all the variable names and types
   //         separated by a colon character :
   //         The variable name and the variable type are separated by a slash (/).
   //         The variable type may be 0,1 or 2 characters. If no type is given,
   //         the type of the variable is assumed to be the same as the previous
   //         variable. If the first variable does not have a type, it is assumed
   //         of type F by default. The list of currently supported types is given below:
   //            - C : a character string terminated by the 0 character
   //            - B : an 8 bit signed integer (Char_t)
   //            - b : an 8 bit unsigned integer (UChar_t)
   //            - S : a 16 bit signed integer (Short_t)
   //            - s : a 16 bit unsigned integer (UShort_t)
   //            - I : a 32 bit signed integer (Int_t)
   //            - i : a 32 bit unsigned integer (UInt_t)
   //            - F : a 32 bit floating point (Float_t)
   //            - D : a 64 bit floating point (Double_t)
   //            - L : a 64 bit signed integer (Long64_t)
   //            - l : a 64 bit unsigned integer (ULong64_t)
   //
   //         By default, a variable will be copied to the buffer with the number of
   //         bytes specified in the type descriptor character. However, if the type
   //         consists of 2 characters, the second character is an integer that
   //         specifies the number of bytes to be used when copying the variable
   //         to the output buffer. Example:
   //             X         ; variable X, type Float_t
   //             Y/I       : variable Y, type Int_t
   //             Y/I2      ; variable Y, type Int_t converted to a 16 bits integer
   //
   //         Arrays of values are supported with the following syntax:
   //         If leaf name has the form var[nelem], where nelem is alphanumeric, then
   //         If nelem is a leaf name, it is used as the variable size of the array.
   //              The leaf refered to by neleme **MUST** be an int (/I).
   //         If leaf name has the form var[nelem], where nelem is a digit, then
   //            it is used as the fixed size of the array.
   //         If leaf name has the form of a multi dimenantion array (eg var[nelem][nelem2])
   //            where nelem and nelem2 are digits) then
   //            it is used as a 2 dimensional array of fixed size.
   //         Any of other form is not supported.
   //
   //   See an example of a Branch definition in the TTree constructor.
   //
   //   Note that in case the data type is an object, this branch can contain
   //   only this object.
   //
   //    Note that this function is invoked by TTree::Branch
   //
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Init(name,leaflist,compress);
}

void TBranch::Init(const char* name, const char* leaflist, Int_t compress)
{
   // Initialization routine called from the constructor.  This should NOT be made virtual.

   SetBit(TBranch::kDoNotUseBufferMap);
   if ((compress == -1) && fTree->GetDirectory()) {
      TFile* bfile = fTree->GetDirectory()->GetFile();
      if (bfile) {
         fCompress = bfile->GetCompressionLevel();
      }
   }

   fBasketBytes = new Int_t[fMaxBaskets];
   fBasketEntry = new Long64_t[fMaxBaskets];
   fBasketSeek  = new Long64_t[fMaxBaskets];

   for (Int_t i = 0; i < fMaxBaskets; ++i) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i] = 0;
   }

   //
   // Decode the leaflist (search for : as separator).
   //

   char* nameBegin = const_cast<char*>(leaflist);
   Int_t offset = 0;
   // FIXME: Make these string streams instead.
   char* leafname = new char[640];
   char* leaftype = new char[320];
   // Note: The default leaf type is a float.
   strlcpy(leaftype, "F",320);
   char* pos = const_cast<char*>(leaflist);
   const char* leaflistEnd = leaflist + strlen(leaflist);
   for (; pos <= leaflistEnd; ++pos) {
      // -- Scan leaf specification and create leaves.
      if ((*pos == ':') || (*pos == 0)) {
         // -- Reached end of a leaf spec, create a leaf.
         Int_t lenName = pos - nameBegin;
         char* ctype = 0;
         if (lenName) {
            strncpy(leafname, nameBegin, lenName);
            leafname[lenName] = 0;
            ctype = strstr(leafname, "/");
            if (ctype) {
               *ctype = 0;
               strlcpy(leaftype, ctype + 1,320);
            }
         }
         if (lenName == 0 || ctype == leafname) {
            Warning("TBranch","No name was given to the leaf number '%d' in the leaflist of the branch '%s'.",fNleaves,name);
            snprintf(leafname,640,"__noname%d",fNleaves);
         }
         TLeaf* leaf = 0;
         if (*leaftype == 'C') {
            leaf = new TLeafC(this, leafname, leaftype);
         } else if (*leaftype == 'O') {
            leaf = new TLeafO(this, leafname, leaftype);
         } else if (*leaftype == 'B') {
            leaf = new TLeafB(this, leafname, leaftype);
         } else if (*leaftype == 'b') {
            leaf = new TLeafB(this, leafname, leaftype);
            leaf->SetUnsigned();
         } else if (*leaftype == 'S') {
            leaf = new TLeafS(this, leafname, leaftype);
         } else if (*leaftype == 's') {
            leaf = new TLeafS(this, leafname, leaftype);
            leaf->SetUnsigned();
         } else if (*leaftype == 'I') {
            leaf = new TLeafI(this, leafname, leaftype);
         } else if (*leaftype == 'i') {
            leaf = new TLeafI(this, leafname, leaftype);
            leaf->SetUnsigned();
         } else if (*leaftype == 'F') {
            leaf = new TLeafF(this, leafname, leaftype);
         } else if (*leaftype == 'f') {
            leaf = new TLeafF(this, leafname, leaftype);
         } else if (*leaftype == 'L') {
            leaf = new TLeafL(this, leafname, leaftype);
         } else if (*leaftype == 'l') {
            leaf = new TLeafL(this, leafname, leaftype);
            leaf->SetUnsigned();
         } else if (*leaftype == 'D') {
            leaf = new TLeafD(this, leafname, leaftype);
         } else if (*leaftype == 'd') {
            leaf = new TLeafD(this, leafname, leaftype);
         }
         if (!leaf) {
            Error("TLeaf", "Illegal data type for %s/%s", name, leaflist);
            delete[] leaftype;
            delete [] leafname;
            MakeZombie();
            return;
         }
         if (leaf->IsZombie()) {
            delete leaf;
            leaf = 0;
            Error("TBranch", "Illegal leaf: %s/%s", name, leaflist);
            delete [] leafname;
            delete[] leaftype;
            MakeZombie();
            return;
         }
         leaf->SetBranch(this);
         leaf->SetAddress((char*) (fAddress + offset));
         leaf->SetOffset(offset);
         if (leaf->GetLeafCount()) {
            // -- Leaf is a varying length array, we need an offset array.
            fEntryOffsetLen = 1000;
         }
         if (leaf->InheritsFrom(TLeafC::Class())) {
            // -- Leaf is a character string, we need an offset array.
            fEntryOffsetLen = 1000;
         }
         ++fNleaves;
         fLeaves.Add(leaf);
         fTree->GetListOfLeaves()->Add(leaf);
         if (*pos == 0) {
            // -- We reached the end of the leaf specification.
            break;
         }
         nameBegin = pos + 1;
         offset += leaf->GetLenType() * leaf->GetLen();
      }
   }
   delete[] leafname;
   leafname = 0;
   delete[] leaftype;
   leaftype = 0;

}

//______________________________________________________________________________
TBranch::~TBranch()
{
   // Destructor.

   delete fBrowsables;
   fBrowsables = 0;

   // Note: We do *not* have ownership of the buffer.
   fEntryBuffer = 0;

   delete [] fBasketSeek;
   fBasketSeek  = 0;

   delete [] fBasketEntry;
   fBasketEntry = 0;

   delete [] fBasketBytes;
   fBasketBytes = 0;

   fBaskets.Delete();
   fNBaskets = 0;
   fCurrentBasket = 0;
   fFirstBasketEntry = -1;
   fNextBasketEntry = -1;

   // Remove our leaves from our tree's list of leaves.
   if (fTree) {
      TObjArray* lst = fTree->GetListOfLeaves();
      if (lst && lst->GetLast()!=-1) {
         lst->RemoveAll(&fLeaves);
      }
   }
   // And delete our leaves.
   fLeaves.Delete();

   fBranches.Delete();

   // If we are in a directory and that directory is not the same
   // directory that our tree is in, then try to find an open file
   // with the name fFileName.  If we find one, delete that file.
   // We are attempting to close any alternate file which we have
   // been directed to write our baskets to.
   // FIXME: We make no attempt to check if someone else might be
   //        using this file.  This is very user hostile.  A violation
   //        of the principle of least surprises.
   //
   // Warning. Must use FindObject by name instead of fDirectory->GetFile()
   // because two branches may point to the same file and the file
   // may have already been deleted in the previous branch.
   if (fDirectory && (!fTree || fDirectory != fTree->GetDirectory())) {
      TString bFileName( GetRealFileName() );

      TFile* file = (TFile*)gROOT->GetListOfFiles()->FindObject(bFileName);
      if (file){
         file->Close();
         delete file;
         file = 0;
      }
   }

   fTree = 0;
   fDirectory = 0;
}

//______________________________________________________________________________
void TBranch::AddBasket(TBasket& b, Bool_t ondisk, Long64_t startEntry)
{
   // Add the basket to this branch.

   // Warning: if the basket are not 'flushed/copied' in the same
   // order as they were created, this will induce a slow down in
   // the insert (since we'll need to move all the record that are
   // entere 'too early').
   // Warning we also assume that the __current__ write basket is
   // not present (aka has been removed).

   TBasket *basket = &b;

   basket->SetBranch(this);

   if (fWriteBasket >= fMaxBaskets) {
      ExpandBasketArrays();
   }
   Int_t where = fWriteBasket;

   if (where && startEntry < fBasketEntry[where-1]) {
      // Need to find the right location and move the possible baskets

      if (!ondisk) {
         Warning("AddBasket","The assumption that out-of-order basket only comes from disk based ntuple is false.");
      }

      if (startEntry < fBasketEntry[0]) {
         where = 0;
      } else {
         for(Int_t i=fWriteBasket-1; i>=0; --i) {
            if (fBasketEntry[i] < startEntry) {
               where = i+1;
               break;
            } else if (fBasketEntry[i] == startEntry) {
               Error("AddBasket","An out-of-order basket matches the entry number of an existing basket.");
            }
         }
      }

      if (where < fWriteBasket) {
         // We shall move the content of the array
         for (Int_t j=fWriteBasket; j > where; --j) {
            fBasketEntry[j] = fBasketEntry[j-1];
            fBasketBytes[j] = fBasketBytes[j-1];
            fBasketSeek[j]  = fBasketSeek[j-1];
         }
      }
   }
   fBasketEntry[where] = startEntry;

   if (ondisk) {
      fBasketBytes[where] = basket->GetNbytes();  // not for in mem
      fBasketSeek[where] = basket->GetSeekKey();  // not for in mem
      fBaskets.AddAtAndExpand(0,fWriteBasket);
      ++fWriteBasket;
   } else {
      ++fNBaskets;
      fBaskets.AddAtAndExpand(basket,fWriteBasket);
      fTree->IncrementTotalBuffers(basket->GetBufferSize());
   }

   fEntries += basket->GetNevBuf();
   fEntryNumber += basket->GetNevBuf();
   if (ondisk) {
      fTotBytes += basket->GetObjlen() + basket->GetKeylen() ;
      fZipBytes += basket->GetNbytes();
      fTree->AddTotBytes(basket->GetObjlen() + basket->GetKeylen());
      fTree->AddZipBytes(basket->GetNbytes());
   }
}

//______________________________________________________________________________
void TBranch::AddLastBasket(Long64_t startEntry)
{
   // Add the start entry of the write basket (not yet created)

   if (fWriteBasket >= fMaxBaskets) {
      ExpandBasketArrays();
   }
   Int_t where = fWriteBasket;

   if (where && startEntry < fBasketEntry[where-1]) {
      // Need to find the right location and move the possible baskets

      Fatal("AddBasket","The last basket must have the highest entry number (%s/%lld/%d).",GetName(),startEntry,fWriteBasket);

   }
   fBasketEntry[where] = startEntry;
   fBaskets.AddAtAndExpand(0,fWriteBasket);
}

//______________________________________________________________________________
void TBranch::Browse(TBrowser* b)
{
   // Browser interface.

   if (fNleaves > 1) {
      fLeaves.Browse(b);
   } else {
      // Get the name and strip any extra brackets
      // in order to get the full arrays.
      TString name = GetName();
      Int_t pos = name.First('[');
      if (pos!=kNPOS) name.Remove(pos);

      GetTree()->Draw(name, "", b ? b->GetDrawOption() : "");
      if (gPad) gPad->Update();
   }
}

 //______________________________________________________________________________
void TBranch::DeleteBaskets(Option_t* option)
{
   // Loop on all branch baskets. If the file where branch buffers reside is
   // writable, free the disk space associated to the baskets of the branch,
   // then call Reset(). If the option contains "all", delete also the baskets
   // for the subbranches.
   // The branch is reset.
   // NOTE that this function must be used with extreme care. Deleting branch baskets
   // fragments the file and may introduce inefficiencies when adding new entries
   // in the Tree or later on when reading the Tree.

   TString opt = option;
   opt.ToLower();
   TFile *file = GetFile(0);

   if(fDirectory && (fDirectory != gROOT) && fDirectory->IsWritable()) {
      for(Int_t i=0; i<fWriteBasket; i++) {
         if (fBasketSeek[i]) file->MakeFree(fBasketSeek[i],fBasketSeek[i]+fBasketBytes[i]-1);
      }
   }

   // process subbranches
   if (opt.Contains("all")) {
      TObjArray *lb = GetListOfBranches();
      Int_t nb = lb->GetEntriesFast();
      for (Int_t j = 0; j < nb; j++) {
         TBranch* branch = (TBranch*) lb->UncheckedAt(j);
         if (branch) branch->DeleteBaskets("all");
      }
   }
   DropBaskets("all");
   Reset();
}

//______________________________________________________________________________
void TBranch::DropBaskets(Option_t* options)
{
   // Loop on all branch baskets. Drop all baskets from memory except readbasket.
   // If the option contains "all", drop all baskets including
   // read- and write-baskets (unless they are not stored individually on disk).
   // The option "all" also lead to DropBaskets being called on the sub-branches.

   Bool_t all = kFALSE;
   if (options && options[0]) {
      TString opt = options;
      opt.ToLower();
      if (opt.Contains("all")) all = kTRUE;
   }

   TBasket *basket;
   Int_t nbaskets = fBaskets.GetEntriesFast();

   if ( (fNBaskets>1) || all ) {
      //slow case
      for (Int_t i=0;i<nbaskets;i++) {
         basket = (TBasket*)fBaskets.UncheckedAt(i);
         if (!basket) continue;
         if ((i == fReadBasket || i == fWriteBasket) && !all) continue;
         if (fBasketBytes[i]==0) continue; // Since it is not on file, we can read it back.
         basket->DropBuffers();
         --fNBaskets;
         fBaskets.RemoveAt(i);
         if (basket == fCurrentBasket) {
            fCurrentBasket    = 0;
            fFirstBasketEntry = -1;
            fNextBasketEntry  = -1;
         }
         delete basket;
      }

      // process subbranches
      if (all) {
         TObjArray *lb = GetListOfBranches();
         Int_t nb = lb->GetEntriesFast();
         for (Int_t j = 0; j < nb; j++) {
            TBranch* branch = (TBranch*) lb->UncheckedAt(j);
            if (!branch) continue;
            branch->DropBaskets("all");
         }
      }
   } else {
      //fast case
      if (nbaskets > 0) {
         Int_t i = fBaskets.GetLast();
         basket = (TBasket*)fBaskets.UncheckedAt(i);
         if (basket && fBasketBytes[i]!=0) {
            basket->DropBuffers();
            if (basket == fCurrentBasket) {
               fCurrentBasket    = 0;
               fFirstBasketEntry = -1;
               fNextBasketEntry  = -1;
            }            
            delete basket;
            fBaskets.AddAt(0,i);
            fBaskets.SetLast(-1);
            fNBaskets = 0;
         }
      }
   }

}

//______________________________________________________________________________
void TBranch::ExpandBasketArrays()
{
   // Increase BasketEntry buffer of a minimum of 10 locations
   // and a maximum of 50 per cent of current size.

   Int_t newsize = TMath::Max(10,Int_t(1.5*fMaxBaskets));
   fBasketBytes  = TStorage::ReAllocInt(fBasketBytes, newsize, fMaxBaskets);
   fBasketEntry  = (Long64_t*)TStorage::ReAlloc(fBasketEntry,
                                                newsize*sizeof(Long64_t),fMaxBaskets*sizeof(Long64_t));
   fBasketSeek   = (Long64_t*)TStorage::ReAlloc(fBasketSeek,
                                                newsize*sizeof(Long64_t),fMaxBaskets*sizeof(Long64_t));

   fMaxBaskets   = newsize;

   fBaskets.Expand(newsize);

   for (Int_t i=fWriteBasket;i<fMaxBaskets;i++) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i]  = 0;
   }
}

//______________________________________________________________________________
Int_t TBranch::Fill()
{
   // Loop on all leaves of this branch to fill Basket buffer.
   //
   // The function returns the number of bytes committed to the memory basket.
   // If a write error occurs, the number of bytes returned is -1.
   // If no data are written, because e.g. the branch is disabled,
   // the number of bytes returned is 0.
   //

   if (TestBit(kDoNotProcess)) {
      return 0;
   }

   TBasket* basket = GetBasket(fWriteBasket);
   if (!basket) {
      basket = fTree->CreateBasket(this); //  create a new basket
      if (!basket) return 0;
      ++fNBaskets;
      fBaskets.AddAtAndExpand(basket,fWriteBasket);
   }
   TBuffer* buf = basket->GetBufferRef();

   // Fill basket buffer.

   Int_t nsize  = 0;

   if (buf->IsReading()) {
      basket->SetWriteMode();
   }

   buf->ResetMap();

   Int_t lold = buf->Length();
   Int_t objectStart = 0;
   Int_t last = 0;
   Int_t lnew = 0;
   Int_t nbytes = 0;

   if (fEntryBuffer) {
      if (fEntryBuffer->IsA() == TMessage::Class()) {
         objectStart = 8;
      }
      if (fEntryBuffer->TestBit(TBufferFile::kNotDecompressed)) {
         // The buffer given as input as not been decompressed.
         if (basket->GetNevBuf()) {
            // If the basket already contains entry we need to close it
            // out. (This is because we can only transfer full compressed
            // buffer)
            WriteBasket(basket,fWriteBasket);
            // And restart from scratch
            return Fill();
         }
         Int_t startpos = fEntryBuffer->Length();
         fEntryBuffer->SetBufferOffset(0);
         static TBasket toread_fLast;
         fEntryBuffer->SetReadMode();
         toread_fLast.Streamer(*fEntryBuffer);
         fEntryBuffer->SetWriteMode();
         last = toread_fLast.GetLast();
         // last now contains the decompressed number of bytes.
         fEntryBuffer->SetBufferOffset(startpos);
         buf->SetBufferOffset(0);
         buf->SetBit(TBufferFile::kNotDecompressed);
         basket->Update(lold);
      } else {
         // We are required to copy starting at the version number (so not
         // including the class name.
         // See if byte count is here, if not it class still be a newClass
         const UInt_t kNewClassTag = 0xFFFFFFFF;
         const UInt_t kByteCountMask = 0x40000000;  // OR the byte count with this
         UInt_t tag = 0;
         UInt_t startpos = fEntryBuffer->Length();
         fEntryBuffer->SetBufferOffset(objectStart);
         *fEntryBuffer >> tag;
         if (tag & kByteCountMask) {
            *fEntryBuffer >> tag;
         }
         if (tag == kNewClassTag) {
            UInt_t maxsize = 256;
            char* s = new char[maxsize];
            Int_t name_start = fEntryBuffer->Length();
            fEntryBuffer->ReadString(s, maxsize); // Reads at most maxsize - 1 characters, plus null at end.
            while (strlen(s) == (maxsize - 1)) {
               // The classname is too large, try again with a large buffer.
               fEntryBuffer->SetBufferOffset(name_start);
               maxsize *= 2;
               delete[] s;
               s = new char[maxsize];
               fEntryBuffer->ReadString(s, maxsize); // Reads at most maxsize - 1 characters, plus null at end
            }
         } else {
            fEntryBuffer->SetBufferOffset(objectStart);
         }
         objectStart = fEntryBuffer->Length();
         fEntryBuffer->SetBufferOffset(startpos);
         basket->Update(lold, objectStart - fEntryBuffer->GetBufferDisplacement());
      }
      fEntries++;
      fEntryNumber++;
      UInt_t len = 0;
      UInt_t startpos = fEntryBuffer->Length();
      if (startpos > UInt_t(objectStart)) {
         // We assume this buffer have just been directly filled
         // the current position in the buffer indicates the end of the object!
         len = fEntryBuffer->Length() - objectStart;
      } else {
         // The buffer have been acquired either via TSocket or via
         // TBuffer::SetBuffer(newloc,newsize)
         // Only the actual size of the memory buffer gives us an hint about where
         // the object ends.
         len = fEntryBuffer->BufferSize() - objectStart;
      }
      buf->WriteBuf(fEntryBuffer->Buffer() + objectStart, len);
      if (fEntryBuffer->TestBit(TBufferFile::kNotDecompressed)) {
         // The original buffer came pre-compressed and thus the buffer Length
         // does not really show the really object size
         // lnew = nbytes = basket->GetLast();
         nbytes = last;
         lnew = last;
      } else {
         lnew = buf->Length();
         nbytes = lnew - lold;
      }
   } else {
      basket->Update(lold);
      ++fEntries;
      ++fEntryNumber;
      FillLeaves(*buf);
      if (buf->GetMapCount()) {
         // The map is used.
         ResetBit(TBranch::kDoNotUseBufferMap);
      }
      lnew = buf->Length();
      nbytes = lnew - lold;
   }

   if (fEntryOffsetLen) {
      Int_t nevbuf = basket->GetNevBuf();
      // Total size in bytes of EntryOffset table.
      nsize = nevbuf * sizeof(Int_t);
   } else {
      if (!basket->GetNevBufSize()) {
         basket->SetNevBufSize(nbytes);
      }
   }

   // Should we create a new basket?
   // fSkipZip force one entry per buffer (old stuff still maintained for CDF)
   // Transfer full compressed buffer only

   if ((fSkipZip && (lnew >= TBuffer::kMinimalSize)) || (buf->TestBit(TBufferFile::kNotDecompressed)) || ((lnew + (2 * nsize) + nbytes) >= fBasketSize)) {
      if (fTree->TestBit(TTree::kCircular)) {
         return nbytes;
      }
      Int_t nout = WriteBasket(basket,fWriteBasket);
      return (nout >= 0) ? nbytes : -1;
   }
   return nbytes;
}

//______________________________________________________________________________
void TBranch::FillLeaves(TBuffer& b)
{
   // Fill each of the leaf of the branch.

   for (Int_t i = 0; i < fNleaves; ++i) {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      leaf->FillBasket(b);
   }
}

//______________________________________________________________________________
TBranch* TBranch::FindBranch(const char* name)
{
   // -- Find the immediate sub-branch with passed name.

   // We allow the user to pass only the last dotted component of the name.
   std::string longnm;
   longnm.reserve(fName.Length()+strlen(name)+3);
   longnm = fName.Data();
   if (longnm[longnm.length()-1]==']') {
      std::size_t dim = longnm.find_first_of("[");
      if (dim != std::string::npos) {
         longnm.erase(dim);
      }
   }
   if (longnm[longnm.length()-1] != '.') {
      longnm += '.';
   }
   longnm += name;
   UInt_t namelen = strlen(name);

   Int_t nbranches = fBranches.GetEntries();
   TBranch* branch = 0;
   for(Int_t i = 0; i < nbranches; ++i) {
      branch = (TBranch*) fBranches.UncheckedAt(i);

      const char *brname = branch->fName.Data();
      UInt_t brlen = branch->fName.Length();
      if (brname[brlen-1]==']') {
         const char *dim = strchr(brname,'[');
         if (dim) {
            brlen = dim - brname;
         }
      }
      if (namelen == brlen /* same effective size */
          && strncmp(name,brname,brlen) == 0) {
         return branch;
      }
      if (brlen == (size_t)longnm.length()
          && strncmp(longnm.c_str(),brname,brlen) == 0) {
         return branch;
      }
   }
   return 0;
}

//______________________________________________________________________________
TLeaf* TBranch::FindLeaf(const char* searchname)
{
   // -- Find the leaf corresponding to the name 'searchname'.

   TString leafname;
   TString leaftitle;
   TString longname;
   TString longtitle;

   // We allow the user to pass only the last dotted component of the name.
   TIter next(GetListOfLeaves());
   TLeaf* leaf = 0;
   while ((leaf = (TLeaf*) next())) {
      leafname = leaf->GetName();
      Ssiz_t dim = leafname.First('[');
      if (dim >= 0) leafname.Remove(dim);

      if (leafname == searchname) return leaf;

      // The leaf element contains the branch name in its name, let's use the title.
      leaftitle = leaf->GetTitle();
      dim = leaftitle.First('[');
      if (dim >= 0) leaftitle.Remove(dim);

      if (leaftitle == searchname) return leaf;

      TBranch* branch = leaf->GetBranch();
      if (branch) {
        longname.Form("%s.%s",branch->GetName(),leafname.Data());
         dim = longname.First('[');
         if (dim>=0) longname.Remove(dim);
         if (longname == searchname) return leaf;

         // The leaf element contains the branch name in its name.
         longname.Form("%s.%s",branch->GetName(),searchname);
         if (longname==leafname) return leaf;

         longtitle.Form("%s.%s",branch->GetName(),leaftitle.Data());
         dim = longtitle.First('[');
         if (dim>=0) longtitle.Remove(dim);
         if (longtitle == searchname) return leaf;

         // The following is for the case where the branch is only
         // a sub-branch.  Since we do not see it through
         // TTree::GetListOfBranches, we need to see it indirectly.
         // This is the less sturdy part of this search ... it may
         // need refining ...
         if (strstr(searchname, ".") && !strcmp(searchname, branch->GetName())) return leaf;
      }
   }
   return 0;
}


//______________________________________________________________________________
Int_t TBranch::FlushBaskets()
{
   // Flush to disk all the baskets of this branch and any of subbranches.
   // Return the number of bytes written or -1 in case of write error.

   UInt_t nerror = 0;
   Int_t nbytes = 0;

   Int_t maxbasket = fWriteBasket + 1;
   // The following protection is not necessary since we should always
   // have fWriteBasket < fBasket.GetSize()
   //if (fBaskets.GetSize() < maxbasket) {
   //   maxbasket = fBaskets.GetSize();
   //}
   for(Int_t i=0; i != maxbasket; ++i) {
      if (fBaskets.UncheckedAt(i)) {
         Int_t nwrite = FlushOneBasket(i);
         if (nwrite<0) {
            ++nerror;
         } else {
            nbytes += nwrite;
         }
      }
   }
   Int_t len = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < len; ++i) {
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      if (!branch) {
         continue;
      }
      Int_t nwrite = branch->FlushBaskets();
      if (nwrite<0) {
         ++nerror;
      } else {
         nbytes += nwrite;
      }
   }
   if (nerror) {
      return -1;
   } else {
      return nbytes;
   }
}

//______________________________________________________________________________
Int_t TBranch::FlushOneBasket(UInt_t ibasket)
{
   // If we have a write basket in memory and it contains some entries and
   // has not yet been written to disk, we write it and delete it from memory.
   // Return the number of bytes written;

   Int_t nbytes = 0;
   if (fDirectory && fBaskets.GetEntries()) {
      TBasket *basket = (TBasket*)fBaskets.UncheckedAt(ibasket);

      if (basket) {
         if (basket->GetNevBuf()
             && fBasketSeek[ibasket]==0) {
            // If the basket already contains entry we need to close it out.
            // (This is because we can only transfer full compressed buffer)

            if (basket->GetBufferRef()->IsReading()) {
               basket->SetWriteMode();
            }
            nbytes = WriteBasket(basket,ibasket);

         } else {
            // If the basket is empty or has already been written.
            if ((Int_t)ibasket==fWriteBasket) {
               // Nothing to do.
            } else {
               basket->DropBuffers();
               if (basket == fCurrentBasket) {
                  fCurrentBasket    = 0;
                  fFirstBasketEntry = -1;
                  fNextBasketEntry  = -1;
               }               
               delete basket;
               --fNBaskets;
               fBaskets[ibasket] = 0;
            }
          }
      }
   }
   return nbytes;
}

//______________________________________________________________________________
TBasket* TBranch::GetBasket(Int_t basketnumber)
{
   //*-*-*-*-*Return pointer to basket basketnumber in this Branch*-*-*-*-*-*
   //*-*      ====================================================

   static Int_t nerrors = 0;

      // reference to an existing basket in memory ?
   if (basketnumber <0 || basketnumber > fWriteBasket) return 0;
   TBasket *basket = (TBasket*)fBaskets.UncheckedAt(basketnumber);
   if (basket) return basket;
   if (basketnumber == fWriteBasket) return 0;

   // create/decode basket parameters from buffer
   TFile *file = GetFile(0);
   if (file == 0) {
      return 0;
   }
   basket = GetFreshBasket();

   // fSkipZip is old stuff still maintained for CDF
   if (fSkipZip) basket->SetBit(TBufferFile::kNotDecompressed);
   if (fBasketBytes[basketnumber] == 0) {
      fBasketBytes[basketnumber] = basket->ReadBasketBytes(fBasketSeek[basketnumber],file);
   }
   //add branch to cache (if any)
   TFileCacheRead *pf = file->GetCacheRead();
   if (pf){
      if (pf->IsLearning()) pf->AddBranch(this);
      if (fSkipZip) pf->SetSkipZip();
   }

   //now read basket
   Int_t badread = basket->ReadBasketBuffers(fBasketSeek[basketnumber],fBasketBytes[basketnumber],file);
   if (badread || basket->GetSeekKey() != fBasketSeek[basketnumber]) {
      nerrors++;
      if (nerrors > 10) return 0;
      if (nerrors == 10) {
         printf(" file probably overwritten: stopping reporting error messages\n");
         if (fBasketSeek[basketnumber] > 2000000000) {
            printf("===>File is more than 2 Gigabytes\n");
            return 0;
         }
         if (fBasketSeek[basketnumber] > 1000000000) {
            printf("===>Your file is may be bigger than the maximum file size allowed on your system\n");
            printf("    Check your AFS maximum file size limit for example\n");
            return 0;
         }
      }
      Error("GetBasket","File: %s at byte:%lld, branch:%s, entry:%lld, badread=%d, nerrors=%d, basketnumber=%d",file->GetName(),basket->GetSeekKey(),GetName(),fReadEntry,badread,nerrors,basketnumber);
      return 0;
   }

   ++fNBaskets;
   fBaskets.AddAt(basket,basketnumber);
   return basket;
}

//______________________________________________________________________________
Long64_t TBranch::GetBasketSeek(Int_t basketnumber) const
{
   //*-*-*-*-*Return address of basket in the file*-*-*-*-*-*
   //*-*      ====================================

   if (basketnumber <0 || basketnumber > fWriteBasket) return 0;
   return fBasketSeek[basketnumber];
}

//______________________________________________________________________________
TList* TBranch::GetBrowsables() {
   // Returns (and, if 0, creates) browsable objects for this branch
   // See TVirtualBranchBrowsable::FillListOfBrowsables.
   if (fBrowsables) return fBrowsables;
   fBrowsables=new TList();
   TVirtualBranchBrowsable::FillListOfBrowsables(*fBrowsables, this);
   return fBrowsables;
}

//______________________________________________________________________________
const char * TBranch::GetClassName() const 
{
   // Return the name of the user class whose content is stored in this branch,
   // if any.  If this branch was created using the 'leaflist' technique, this
   // function returns an empty string.

   return "";
}

//______________________________________________________________________________
const char* TBranch::GetIconName() const
{
   // Return icon name depending on type of branch.

   if (IsFolder())
      return "TBranchElement-folder";
   else
      return "TBranchElement-leaf";
}

//______________________________________________________________________________
Int_t TBranch::GetEntry(Long64_t entry, Int_t getall)
{
   // Read all leaves of entry and return total number of bytes read.
   //
   // The input argument "entry" is the entry number in the current tree.
   // In case of a TChain, the entry number in the current Tree must be found
   // before calling this function. For example:
   //
   //     TChain* chain = ...;
   //     Long64_t localEntry = chain->LoadTree(entry);
   //     branch->GetEntry(localEntry);
   //
   // The function returns the number of bytes read from the input buffer.
   // If entry does not exist, the function returns 0.
   // If an I/O error occurs, the function returns -1.
   //
   // See IMPORTANT REMARKS in TTree::GetEntry.
   //

   Bool_t enabled = !TestBit(kDoNotProcess) || getall;
   TBasket *basket; // will be initialized in the if/then clauses.
   Long64_t first;
   if (0 && R__likely(enabled && fFirstBasketEntry <= entry && entry < fNextBasketEntry)) {
      // We have found the basket containing this entry.
      // make sure basket buffers are in memory.
      basket = fCurrentBasket;
      first = fFirstBasketEntry;
   } else {
      if (!enabled) {
         return 0;
      }
      if ((entry < fFirstEntry) || (entry >= fEntryNumber)) {
         return 0;
      }
      first = fFirstBasketEntry;
      Long64_t last = fNextBasketEntry - 1;
      // Are we still in the same ReadBasket?
      if ((entry < first) || (entry > last)) {
         fReadBasket = TMath::BinarySearch(fWriteBasket + 1, fBasketEntry, entry);
         if (fReadBasket < 0) {
            fNextBasketEntry = -1;
            Error("In the branch %s, no basket contains the entry %d\n", GetName(), entry);
            return -1;
         }
         if (fReadBasket == fWriteBasket) {
            fNextBasketEntry = fEntryNumber;
         } else {
            fNextBasketEntry = fBasketEntry[fReadBasket+1];
         }
         first = fFirstBasketEntry = fBasketEntry[fReadBasket];
      }
      // We have found the basket containing this entry.
      // make sure basket buffers are in memory.
      basket = (TBasket*) fBaskets.UncheckedAt(fReadBasket);
      if (!basket) {
         basket = GetBasket(fReadBasket);
         if (!basket) {
            fCurrentBasket = 0;
            fFirstBasketEntry = -1; 
            fNextBasketEntry = -1;
            return -1;
         }
      }
      fCurrentBasket = basket;
   }
   basket->PrepareBasket(entry);
   TBuffer* buf = basket->GetBufferRef();
   
   // This test necessary to read very old Root files (NvE).
   if (R__unlikely(!buf)) {
      TFile* file = GetFile(0);
      basket->ReadBasketBuffers(fBasketSeek[fReadBasket], fBasketBytes[fReadBasket], file);
      buf = basket->GetBufferRef();
   }
   // Set entry offset in buffer.
   if (!TestBit(kDoNotUseBufferMap)) {
      buf->ResetMap();
   }
   if (R__unlikely(!buf->IsReading())) {
      basket->SetReadMode();
   }
   Int_t* entryOffset = basket->GetEntryOffset();
   Int_t bufbegin = 0;
   if (entryOffset) {
      bufbegin = entryOffset[entry-first];
      Int_t* displacement = basket->GetDisplacement();
      if (R__unlikely(displacement)) {
         buf->SetBufferDisplacement(displacement[entry-first]);
      }
   } else {
      bufbegin = basket->GetKeylen() + ((entry-first) * basket->GetNevBufSize());
   }
   buf->SetBufferOffset(bufbegin);
   
   // Int_t bufbegin = buf->Length();
   // Remember which entry we are reading.
   fReadEntry = entry;
   (this->*fReadLeaves)(*buf);
   return buf->Length() - bufbegin;
}

//______________________________________________________________________________
Int_t TBranch::GetEntryExport(Long64_t entry, Int_t /*getall*/, TClonesArray* li, Int_t nentries)
{
   // Read all leaves of an entry and export buffers to real objects in a TClonesArray list.
   //
   // Returns total number of bytes read.

   if (TestBit(kDoNotProcess)) {
      return 0;
   }
   if ((entry < 0) || (entry >= fEntryNumber)) {
      return 0;
   }
   Int_t nbytes = 0;
   Long64_t first  = fFirstBasketEntry;
   Long64_t last = fNextBasketEntry - 1;
   // Are we still in the same ReadBasket?
   if ((entry < first) || (entry > last)) {
      fReadBasket = TMath::BinarySearch(fWriteBasket + 1, fBasketEntry, entry);
      if (fReadBasket < 0) {
         fNextBasketEntry = -1;
         Error("In the branch %s, no basket contains the entry %d\n", GetName(), entry);
         return -1;
      }
      if (fReadBasket == fWriteBasket) {
         fNextBasketEntry = fEntryNumber;
      } else {
         fNextBasketEntry = fBasketEntry[fReadBasket+1];
      }
      fFirstBasketEntry = first = fBasketEntry[fReadBasket];
   }

   // We have found the basket containing this entry.
   // Make sure basket buffers are in memory.
   TBasket* basket = GetBasket(fReadBasket);
   fCurrentBasket = basket;
   if (!basket) {
      fFirstBasketEntry = -1; 
      fNextBasketEntry = -1;
      return 0;
   }
   TBuffer* buf = basket->GetBufferRef();
   // Set entry offset in buffer and read data from all leaves.
   if (!buf->IsReading()) {
      basket->SetReadMode();
   }
   Int_t bufbegin = 0;
   Int_t* entryOffset = basket->GetEntryOffset();
   if (entryOffset) {
      bufbegin = entryOffset[entry-first];
   } else {
      bufbegin = basket->GetKeylen() + ((entry - first) * basket->GetNevBufSize());
   }
   buf->SetBufferOffset(bufbegin);
   Int_t* displacement = basket->GetDisplacement();
   if (displacement) {
      buf->SetBufferDisplacement(displacement[entry-first]);
   } else {
      buf->SetBufferDisplacement();
   }
   // Remember which entry we are reading.
   fReadEntry = entry;
   TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(0);
   leaf->ReadBasketExport(*buf, li, nentries);
   nbytes = buf->Length() - bufbegin;
   return nbytes;
}

//______________________________________________________________________________
Int_t TBranch::GetExpectedType(TClass *&expectedClass,EDataType &expectedType)
{
   // Fill expectedClass and expectedType with information on the data type of the 
   // object/values contained in this branch (and thus the type of pointers
   // expected to be passed to Set[Branch]Address
   // return 0 in case of success and > 0 in case of failure.
   
   expectedClass = 0;
   expectedType = kOther_t;
   TLeaf* l = (TLeaf*) GetListOfLeaves()->At(0);
   if (l) {
      expectedType = (EDataType) gROOT->GetType(l->GetTypeName())->GetType();
      return 0;
   } else {
      Error("GetExpectedType", "Did not find any leaves in %s",GetName());
      return 1;
   }
}

//______________________________________________________________________________
TFile* TBranch::GetFile(Int_t mode)
{
   // Return pointer to the file where branch buffers reside, returns 0
   // in case branch buffers reside in the same file as tree header.
   // If mode is 1 the branch buffer file is recreated.

   if (fDirectory) return fDirectory->GetFile();

   // check if a file with this name is in the list of Root files
   TFile *file = (TFile*)gROOT->GetListOfFiles()->FindObject(fFileName.Data());
   if (file) {
      fDirectory = file;
      return file;
   }

   if (fFileName.Length() == 0) return 0;

   TString bFileName( GetRealFileName() );

   // Open file (new file if mode = 1)
   {
      TDirectory::TContext ctxt(0);
      if (mode) file = TFile::Open(bFileName, "recreate");
      else      file = TFile::Open(bFileName);
   }
   if (!file) return 0;
   if (file->IsZombie()) {delete file; return 0;}
   fDirectory = (TDirectory*)file;
   return file;
}

//______________________________________________________________________________
TBasket* TBranch::GetFreshBasket()
{
   // Return a fresh basket by either resusing an existing basket that needs
   // to be drop (according to TTree::MemoryFull) or create a new one.
   
   TBasket *basket = 0;  
   if (GetTree()->MemoryFull(0)) {
      if (fNBaskets==1) {
         // Steal the existing basket
         Int_t oldindex = fBaskets.GetLast();
         basket = (TBasket*)fBaskets.UncheckedAt(oldindex);
         if (!basket) {
            fBaskets.SetLast(-2); // For recalculation of Last.
            oldindex = fBaskets.GetLast();
            basket = (TBasket*)fBaskets.UncheckedAt(oldindex);
         }
         if (basket && fBasketBytes[oldindex]!=0) {
            if (basket == fCurrentBasket) {
               fCurrentBasket    = 0;
               fFirstBasketEntry = -1;
               fNextBasketEntry  = -1;
            }            
            fBaskets.AddAt(0,oldindex);
            fBaskets.SetLast(-1);
            fNBaskets = 0;
         } else {
            basket = fTree->CreateBasket(this);
         }            
      } else if (fNBaskets == 0) {
         // There is nothing to drop!
         basket = fTree->CreateBasket(this);
      } else {
         // Memory is full and there is more than one basket,
         // Let DropBaskets do it job.
         DropBaskets();
         basket = fTree->CreateBasket(this);
      }
   } else {
      basket = fTree->CreateBasket(this);
   }   
   return basket;
}

//______________________________________________________________________________
TLeaf* TBranch::GetLeaf(const char* name) const
{
   //*-*-*-*-*-*Return pointer to the 1st Leaf named name in thisBranch-*-*-*-*-*
   //*-*        =======================================================

   Int_t i;
   for (i=0;i<fNleaves;i++) {
      TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(i);
      if (!strcmp(leaf->GetName(),name)) return leaf;
   }
   return 0;
}

//______________________________________________________________________________
TString TBranch::GetRealFileName() const
{
   if (fFileName.Length()==0) {
      return fFileName;
   }
   TString bFileName = fFileName;

   // check if branch file name is absolute or a URL (e.g. /castor/...,
   // root://host/..., rfio:/path/...)
   char *bname = gSystem->ExpandPathName(fFileName.Data());
   if (!gSystem->IsAbsoluteFileName(bname) && !strstr(bname, ":/") && fTree && fTree->GetCurrentFile()) {

      // if not, get filename where tree header is stored
      const char *tfn = fTree->GetCurrentFile()->GetName();

      // If it is an archive file we need a special treatment
      TUrl arc(tfn);
      if (strlen(arc.GetAnchor()) > 0) {
         arc.SetAnchor(gSystem->BaseName(fFileName));
         bFileName = arc.GetUrl();
      } else {
         // if this is an absolute path or a URL then prepend this path
         // to the branch file name
         char *tname = gSystem->ExpandPathName(tfn);
         if (gSystem->IsAbsoluteFileName(tname) || strstr(tname, ":/")) {
            bFileName = gSystem->DirName(tname);
            bFileName += "/";
            bFileName += fFileName;
         }
         delete [] tname;
      }
   }
   delete [] bname;

   return bFileName;
}

//______________________________________________________________________________
Int_t TBranch::GetRow(Int_t)
{
   //*-*-*-*-*Return all elements of one row unpacked in internal array fValues*-*
   //*-*      =================================================================

   return 1;
}

//______________________________________________________________________________
Bool_t TBranch::GetMakeClass() const
{
   // Return whether this branch is in a mode where the object are decomposed
   // or not (Also known as MakeClass mode).
   
   // Regular TBranch and TBrancObject can not be in makeClass mode

   return kFALSE;
}
   
//______________________________________________________________________________
TBranch* TBranch::GetMother() const
{
   // Get our top-level parent branch in the tree.

   if (fMother) return fMother;

   const TObjArray* array = fTree->GetListOfBranches();
   Int_t n = array->GetEntriesFast();
   for (Int_t i = 0; i < n; ++i) {
      TBranch* branch = (TBranch*) array->UncheckedAt(i);
      TBranch* parent = branch->GetSubBranch(this);
      if (parent) {
         const_cast<TBranch*>(this)->fMother = branch; // We can not yet use the 'mutable' keyword
         return branch;
      }
   }
   return 0;
}

//______________________________________________________________________________
TBranch* TBranch::GetSubBranch(const TBranch* child) const
{
   // Find the parent branch of child.
   // Return 0 if child is not in this branch hierarchy.

   // Handle error condition, if the parameter is us, we cannot find the parent.
   if (this == child) {
      // Note: We cast away any const-ness of "this".
      return (TBranch*) this;
   }

   if (child->fParent) {
      return child->fParent;
   }

   Int_t len = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < len; ++i) {
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      if (!branch) {
         continue;
      }
      if (branch == child) {
         // We are the direct parent of child.
         const_cast<TBranch*>(child)->fParent = (TBranch*)this; // We can not yet use the 'mutable' keyword
         // Note: We cast away any const-ness of "this".
         const_cast<TBranch*>(child)->fParent = (TBranch*)this; // We can not yet use the 'mutable' keyword
         return (TBranch*) this;
      }
      // FIXME: This is a tail-recursion!
      TBranch* parent = branch->GetSubBranch(child);
      if (parent) {
         return parent;
      }
   }
   // We failed to find the parent.
   return 0;
}

//______________________________________________________________________________
Long64_t TBranch::GetTotalSize(Option_t * /*option*/) const
{
   // Return total number of bytes in the branch (including current buffer)

   TObjArray &baskets( const_cast<TObjArray&>(fBaskets) );
   TBasket *writebasket = 0;
   if (fNBaskets == 1) {
      writebasket = (TBasket*)fBaskets.UncheckedAt(fWriteBasket);
      if (writebasket && writebasket->GetNevBuf()==0) {
         baskets[fWriteBasket] = 0;
      } else {
         writebasket = 0;
      }
   }
   TBufferFile b(TBuffer::kWrite,10000);
   TBranch::Class()->WriteBuffer(b,(TBranch*)this);
   if (writebasket) {
      baskets[fWriteBasket] = writebasket;         
   }
   Long64_t totbytes = 0;
   if (fZipBytes > 0) totbytes = fTotBytes;
   return totbytes + b.Length();
}

//______________________________________________________________________________
Long64_t TBranch::GetTotBytes(Option_t *option) const
{
   // Return total number of bytes in the branch (excluding current buffer)
   // if option ="*" includes all sub-branches of this branch too

   Long64_t totbytes = fTotBytes;
   if (!option) return totbytes;
   if (option[0] != '*') return totbytes;
   //scan sub-branches
   Int_t len = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < len; ++i) {
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      if (branch) totbytes += branch->GetTotBytes();
   }
   return totbytes;
}

//______________________________________________________________________________
Long64_t TBranch::GetZipBytes(Option_t *option) const
{
   // Return total number of zip bytes in the branch
   // if option ="*" includes all sub-branches of this branch too

   Long64_t zipbytes = fZipBytes;
   if (!option) return zipbytes;
   if (option[0] != '*') return zipbytes;
   //scan sub-branches
   Int_t len = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < len; ++i) {
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      if (branch) zipbytes += branch->GetZipBytes();
   }
   return zipbytes;
}

//______________________________________________________________________________
Bool_t TBranch::IsAutoDelete() const
{
   // Return kTRUE if an existing object in a TBranchObject must be deleted.
   return TestBit(kAutoDelete);
}

//______________________________________________________________________________
Bool_t TBranch::IsFolder() const
{
   // Return kTRUE if more than one leaf or browsables, kFALSE otherwise.
   if (fNleaves > 1) {
      return kTRUE;
   }
   TList* browsables = const_cast<TBranch*>(this)->GetBrowsables();
   return browsables && browsables->GetSize();
}

//______________________________________________________________________________
void TBranch::KeepCircular(Long64_t maxEntries)
{
   // keep a maximum of fMaxEntries in memory

   Int_t dentries = (Int_t) (fEntries - maxEntries);
   TBasket* basket = (TBasket*) fBaskets.UncheckedAt(0);
   basket->MoveEntries(dentries);
   fEntries = maxEntries;
   fEntryNumber = maxEntries;
   //loop on sub branches
   Int_t nb = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nb; ++i)  {
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      branch->KeepCircular(maxEntries);
   }
}

//______________________________________________________________________________
Int_t TBranch::LoadBaskets()
{
   //  Baskets associated to this branch are forced to be in memory.
   //  You can call TTree::SetMaxVirtualSize(maxmemory) to instruct
   //  the system that the total size of the imported baskets does not
   //  exceed maxmemory bytes.
   //  The function returns the number of baskets that have been put in memory.
   //  This method may be called to force all baskets of one or more branches
   //  in memory when random access to entries in this branch is required.
   //  See also TTree::LoadBaskets to load all baskets of all branches in memory.

   Int_t nimported = 0;
   Int_t nbaskets = fWriteBasket;
   TFile *file = GetFile(0);
   TBasket *basket;
   for (Int_t i=0;i<nbaskets;i++) {
      basket = (TBasket*)fBaskets.UncheckedAt(i);
      if (basket) continue;
      basket = GetFreshBasket();
      if (fBasketBytes[i] == 0) {
         fBasketBytes[i] = basket->ReadBasketBytes(fBasketSeek[i],file);
      }
      Int_t badread = basket->ReadBasketBuffers(fBasketSeek[i],fBasketBytes[i],file);
      if (badread) {
         Error("Loadbaskets","Error while reading basket buffer %d of branch %s",i,GetName());
         return -1;
      }
      ++fNBaskets;
      fBaskets.AddAt(basket,i);
      nimported++;
   }
   return nimported;
}

//______________________________________________________________________________
void TBranch::Print(Option_t*) const
{
   // Print TBranch parameters

   const int kLINEND = 77;
   Float_t cx = 1;

   TString titleContent(GetTitle());
   if ( titleContent == GetName() ) {
      titleContent.Clear();
   }
   
   if (fLeaves.GetEntries() == 1) {
      if (titleContent[titleContent.Length()-2]=='/' && isalpha(titleContent[titleContent.Length()-1])) {
         // The type is already encoded.  Nothing to do.
      } else {
         TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(0);
         if (titleContent.Length()) {
            titleContent.Prepend(" ");
         }
         // titleContent.Append("type: ");
         titleContent.Prepend(leaf->GetTypeName());
      }
   }
   Int_t titleLength = titleContent.Length();

   Int_t aLength = titleLength + strlen(GetName());
   aLength += (aLength / 54 + 1) * 80 + 100;
   if (aLength < 200) aLength = 200;
   char *bline = new char[aLength];

   Long64_t totBytes = GetTotalSize();
   if (fZipBytes) cx = (fTotBytes+0.00001)/fZipBytes;
   if (titleLength) snprintf(bline,aLength,"*Br%5d :%-9s : %-54s *",fgCount,GetName(),titleContent.Data());
   else             snprintf(bline,aLength,"*Br%5d :%-9s : %-54s *",fgCount,GetName()," ");
   if (strlen(bline) > UInt_t(kLINEND)) {
      char *tmp = new char[strlen(bline)+1];
      if (titleLength) strlcpy(tmp, titleContent.Data(),strlen(bline)+1);
      snprintf(bline,aLength,"*Br%5d :%-9s : ",fgCount,GetName());
      int pos = strlen (bline);
      int npos = pos;
      int beg=0, end;
      while (beg < titleLength) {
         for (end=beg+1; end < titleLength-1; end ++)
            if (tmp[end] == ':')  break;
         if (npos + end-beg+1 >= 78) {
            while (npos < kLINEND) {
               bline[pos ++] = ' ';
               npos ++;
            }
            bline[pos ++] = '*';
            bline[pos ++] = '\n';
            bline[pos ++] = '*';
            npos = 1;
            for (; npos < 12; npos ++)
               bline[pos ++] = ' ';
            bline[pos-2] = '|';
         }
         for (int n = beg; n <= end; n ++)
            bline[pos+n-beg] = tmp[n];
         pos += end-beg+1;
         npos += end-beg+1;
         beg = end+1;
      }
      while (npos < kLINEND) {
         bline[pos ++] = ' ';
         npos ++;
      }
      bline[pos ++] = '*';
      bline[pos] = '\0';
      delete[] tmp;
   }
   Printf("%s", bline);
   if (fTotBytes > 2000000000) {
      Printf("*Entries :%lld : Total  Size=%11lld bytes  File Size  = %lld *",fEntries,totBytes,fZipBytes);
   } else {
      if (fZipBytes > 0) {
         Printf("*Entries :%9lld : Total  Size=%11lld bytes  File Size  = %10lld *",fEntries,totBytes,fZipBytes);
      } else {
         if (fWriteBasket > 0) {
            Printf("*Entries :%9lld : Total  Size=%11lld bytes  All baskets in memory   *",fEntries,totBytes);
         } else {
            Printf("*Entries :%9lld : Total  Size=%11lld bytes  One basket in memory    *",fEntries,totBytes);
         }
      }
   }
   Printf("*Baskets :%9d : Basket Size=%11d bytes  Compression= %6.2f     *",fWriteBasket,fBasketSize,cx);
   Printf("*............................................................................*");
   delete [] bline;
   fgCount++;
}

//______________________________________________________________________________
void TBranch::ReadBasket(TBuffer&)
{
   // Loop on all leaves of this branch to read Basket buffer.

   //   fLeaves->ReadBasket(basket);
}

//______________________________________________________________________________
void TBranch::ReadLeavesImpl(TBuffer& b)
{
   // Loop on all leaves of this branch to read Basket buffer.
   
   for (Int_t i = 0; i < fNleaves; ++i) {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      leaf->ReadBasket(b);
   }
}

//______________________________________________________________________________
void TBranch::ReadLeaves0Impl(TBuffer&)
{
   // Loop on all leaves of this branch to read Basket buffer.
   
}

//______________________________________________________________________________
void TBranch::ReadLeaves1Impl(TBuffer& b)
{
   // Loop on all leaves of this branch to read Basket buffer.
   
   ((TLeaf*) fLeaves.UncheckedAt(0))->ReadBasket(b);
}

//______________________________________________________________________________
void TBranch::ReadLeaves2Impl(TBuffer& b)
{
   // Loop on all leaves of this branch to read Basket buffer.
   
   ((TLeaf*) fLeaves.UncheckedAt(0))->ReadBasket(b);
   ((TLeaf*) fLeaves.UncheckedAt(1))->ReadBasket(b);
}

//______________________________________________________________________________
void TBranch::Refresh(TBranch* b)
{
   //  refresh this branch using new information in b
   //  This function is called by TTree::Refresh

   fEntryOffsetLen = b->fEntryOffsetLen;
   fWriteBasket    = b->fWriteBasket;
   fEntryNumber    = b->fEntryNumber;
   fMaxBaskets     = b->fMaxBaskets;
   fEntries        = b->fEntries;
   fTotBytes       = b->fTotBytes;
   fZipBytes       = b->fZipBytes;
   fReadBasket     = 0;
   fReadEntry      = -1;
   fFirstBasketEntry = -1;
   fNextBasketEntry  = -1;
   fCurrentBasket    =  0;
   delete [] fBasketBytes;
   delete [] fBasketEntry;
   delete [] fBasketSeek;
   fBasketBytes = new Int_t[fMaxBaskets];
   fBasketEntry = new Long64_t[fMaxBaskets];
   fBasketSeek  = new Long64_t[fMaxBaskets];
   Int_t i;
   for (i=0;i<fMaxBaskets;i++) {
      fBasketBytes[i] = b->fBasketBytes[i];
      fBasketEntry[i] = b->fBasketEntry[i];
      fBasketSeek[i]  = b->fBasketSeek[i];
   }
   fBaskets.Delete();
   Int_t nbaskets = b->fBaskets.GetSize();
   fBaskets.Expand(nbaskets);
   // If the current fWritebasket is in memory, take it (just swap)
   // from the Tree being read
   TBasket *basket = (TBasket*)b->fBaskets.UncheckedAt(fWriteBasket);
   fBaskets.AddAt(basket,fWriteBasket);
   if (basket) {
      fNBaskets = 1;
      --(b->fNBaskets);
      b->fBaskets.RemoveAt(fWriteBasket);
      basket->SetBranch(this);
   }
}

//______________________________________________________________________________
void TBranch::Reset(Option_t*)
{
   // Reset a Branch.
   //
   // Existing buffers are deleted.
   // Entries, max and min are reset.
   //

   fReadBasket = 0;
   fReadEntry = -1;
   fFirstBasketEntry = -1;
   fNextBasketEntry = -1;
   fCurrentBasket   = 0;
   fWriteBasket = 0;
   fEntries = 0;
   fTotBytes = 0;
   fZipBytes = 0;
   fEntryNumber = 0;

   if (fBasketBytes) {
      for (Int_t i = 0; i < fMaxBaskets; ++i) {
         fBasketBytes[i] = 0;
      }
   }

   if (fBasketEntry) {
      for (Int_t i = 0; i < fMaxBaskets; ++i) {
         fBasketEntry[i] = 0;
      }
   }

   if (fBasketSeek) {
      for (Int_t i = 0; i < fMaxBaskets; ++i) {
         fBasketSeek[i] = 0;
      }
   }

   fBaskets.Delete();
   fNBaskets = 0;
}

//______________________________________________________________________________
void TBranch::ResetAddress()
{
   // Reset the address of the branch.

   fAddress = 0;

   //  Reset last read entry number, we have will had new user object now.
   fReadEntry = -1;
   fFirstBasketEntry = -1;
   fNextBasketEntry  = -1;

   for (Int_t i = 0; i < fNleaves; ++i) {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      leaf->SetAddress(0);
   }

   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i)  {
      TBranch* abranch = (TBranch*) fBranches[i];
      // FIXME: This is a tail recursion.
      abranch->ResetAddress();
   }
}

//______________________________________________________________________________
void TBranch::ResetCount()
{
   // Static function resetting fgCount
   fgCount = 0;
}

//______________________________________________________________________________
void TBranch::SetAddress(void* addr)
{
   // Set address of this branch.
   if (TestBit(kDoNotProcess)) {
      return;
   }
   fReadEntry = -1;
   fFirstBasketEntry = -1;
   fNextBasketEntry  = -1;
   fAddress = (char*) addr;
   for (Int_t i = 0; i < fNleaves; ++i) {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      Int_t offset = leaf->GetOffset();
      if (TestBit(kIsClone)) {
         offset = 0;
      }
      if (fAddress) leaf->SetAddress(fAddress + offset);
      else leaf->SetAddress(0);
   }
}

//______________________________________________________________________________
void TBranch::SetAutoDelete(Bool_t autodel)
{
   // Set the automatic delete bit.
   //
   // This bit is used by TBranchObject::ReadBasket to decide if an object
   // referenced by a TBranchObject must be deleted or not before reading
   // a new entry.
   //
   // If autodel is kTRUE, this existing object will be deleted, a new object
   // created by the default constructor, then read from disk by the streamer.
   //
   // If autodel is kFALSE, the existing object is not deleted.  Root assumes
   // that the user is taking care of deleting any internal object or array
   // (this can be done in the streamer).

   if (autodel) {
      SetBit(kAutoDelete, 1);
   } else {
      SetBit(kAutoDelete, 0);
   }
}

//______________________________________________________________________________
void TBranch::SetBasketSize(Int_t buffsize)
{
   // Set the basket size
   // The function makes sure that the basket size is greater than fEntryOffsetlen

   if (buffsize < 100+fEntryOffsetLen) buffsize = 100+fEntryOffsetLen;
   fBasketSize = buffsize;
   TBasket *basket = (TBasket*)fBaskets[fWriteBasket];
   if (basket) {
      basket->AdjustSize(fBasketSize);
   }
}

//______________________________________________________________________________
void TBranch::SetBufferAddress(TBuffer* buf)
{
   // -- Set address of this branch directly from a TBuffer to avoid streaming.
   //
   // Note: We do not take ownership of the buffer.
   //

   // Check this is possible
   if ( (fNleaves != 1)
       || (strcmp("TLeafObject",fLeaves.UncheckedAt(0)->ClassName())!=0) ) {
      Error("TBranch::SetAddress","Filling from a TBuffer can only be done with a not split object branch.  Request ignored.");
   } else {
      fReadEntry = -1;
      fNextBasketEntry  = -1;
      fFirstBasketEntry = -1;
      // Note: We do not take ownership of the buffer.
      fEntryBuffer = buf;
   }
}

//______________________________________________________________________________
void TBranch::SetCompressionLevel(Int_t level)
{
   //*-*-*-*-*-*-*-*Set the branch/subbranches compression level
   //*-*            ============================================

   fCompress = level;
   Int_t nb = fBranches.GetEntriesFast();

   for (Int_t i=0;i<nb;i++) {
      TBranch *branch = (TBranch*)fBranches.UncheckedAt(i);
      branch->SetCompressionLevel(level);
   }
}

//______________________________________________________________________________
void TBranch::SetEntryOffsetLen(Int_t newdefault, Bool_t updateExisting)
{
   // Update the default value for the branch's fEntryOffsetLen if and only if
   // it was already non zero (and the new value is not zero)
   // If updateExisting is true, also update all the existing branches.

   if (fEntryOffsetLen && newdefault) {
      fEntryOffsetLen = newdefault;
   }
   if (updateExisting) {
      TIter next( GetListOfBranches() );
      TBranch *b;
      while ( ( b = (TBranch*)next() ) ) {
         b->SetEntryOffsetLen( newdefault, kTRUE );
      }
   }
}

//______________________________________________________________________________
void TBranch::SetEntries(Long64_t entries)
{
   // Set the number of entries in this branch.

   fEntries = entries;
   fEntryNumber = entries;
}

//______________________________________________________________________________
void TBranch::SetFile(TFile* file)
{
   // Set file where this branch writes/reads its buffers.
   // By default the branch buffers reside in the file where the
   // Tree was created.
   // If the file name where the tree was created is an absolute
   // path name or an URL (e.g. /castor/... or root://host/...)
   // and if the fname is not an absolute path name or an URL then
   // the path of the tree file is prepended to fname to make the
   // branch file relative to the tree file. In this case one can
   // move the tree + all branch files to a different location in
   // the file system and still access the branch files.
   // The ROOT file will be connected only when necessary.
   // If called by TBranch::Fill (via TBasket::WriteFile), the file
   // will be created with the option "recreate".
   // If called by TBranch::GetEntry (via TBranch::GetBasket), the file
   // will be opened in read mode.
   // To open a file in "update" mode or with a certain compression
   // level, use TBranch::SetFile(TFile *file).

   if (file == 0) file = fTree->GetCurrentFile();
   fDirectory = (TDirectory*)file;
   if (file == fTree->GetCurrentFile()) fFileName = "";
   else                                 fFileName = file->GetName();

   if (file && fCompress == -1) {
      fCompress = file->GetCompressionLevel();      
   }

   // Apply to all existing baskets.
   TIter nextb(GetListOfBaskets());
   TBasket *basket;
   while ((basket = (TBasket*)nextb())) {
      basket->SetParent(file);
   }

   // Apply to sub-branches as well.
   TIter next(GetListOfBranches());
   TBranch *branch;
   while ((branch = (TBranch*)next())) {
      branch->SetFile(file);
   }
}

//______________________________________________________________________________
void TBranch::SetFile(const char* fname)
{
   // Set file where this branch writes/reads its buffers.
   // By default the branch buffers reside in the file where the
   // Tree was created.
   // If the file name where the tree was created is an absolute
   // path name or an URL (e.g. /castor/... or root://host/...)
   // and if the fname is not an absolute path name or an URL then
   // the path of the tree file is prepended to fname to make the
   // branch file relative to the tree file. In this case one can
   // move the tree + all branch files to a different location in
   // the file system and still access the branch files.
   // The ROOT file will be connected only when necessary.
   // If called by TBranch::Fill (via TBasket::WriteFile), the file
   // will be created with the option "recreate".
   // If called by TBranch::GetEntry (via TBranch::GetBasket), the file
   // will be opened in read mode.
   // To open a file in "update" mode or with a certain compression
   // level, use TBranch::SetFile(TFile *file).

   fFileName  = fname;
   fDirectory = 0;

   //apply to sub-branches as well
   TIter next(GetListOfBranches());
   TBranch *branch;
   while ((branch = (TBranch*)next())) {
      branch->SetFile(fname);
   }
}

//______________________________________________________________________________
Bool_t TBranch::SetMakeClass(Bool_t /* decomposeObj */)
{
   // Set the branch in a mode where the object are decomposed
   // (Also known as MakeClass mode).
   // Return whether the setting was possible (it is not possible for
   // TBranch and TBranchObject).
   
   // Regular TBranch and TBrancObject can not be in makeClass mode
   return kFALSE;
}

//______________________________________________________________________________
void TBranch::SetObject(void * /* obj */)
{
   // Set object this branch is pointing to.

   if (TestBit(kDoNotProcess)) {
      return;
   }
   Warning("SetObject","is not supported in TBranch objects");
}

//______________________________________________________________________________
void TBranch::SetStatus(Bool_t status)
{
   // Set branch status to Process or DoNotProcess.
      
   if (status) ResetBit(kDoNotProcess);
   else        SetBit(kDoNotProcess);
}

//_______________________________________________________________________
void TBranch::Streamer(TBuffer& b)
{
   // Stream a class object

   if (b.IsReading()) {
      UInt_t R__s, R__c;
      fTree = gTree;
      fAddress = 0;
      gROOT->SetReadingObject(kTRUE);
      
      // Reset transients.
      SetBit(TBranch::kDoNotUseBufferMap);
      fCurrentBasket    = 0;
      fFirstBasketEntry = -1;
      fNextBasketEntry  = -1;

      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 9) {
         b.ReadClassBuffer(TBranch::Class(), this, v, R__s, R__c);

         if (fWriteBasket>=fBaskets.GetSize()) {
            fBaskets.Expand(fWriteBasket+1);
         }
         fDirectory = 0;
         fNleaves = fLeaves.GetEntriesFast();
         for (Int_t i=0;i<fNleaves;i++) {
            TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(i);
            leaf->SetBranch(this);
         }

         fNBaskets = fBaskets.GetEntries();
         for (Int_t j=fWriteBasket,n=0;j>=0 && n<fNBaskets;--j) {
            TBasket *bk = (TBasket*)fBaskets.UncheckedAt(j);
            if (bk) {
               bk->SetBranch(this);
               GetTree()->IncrementTotalBuffers(bk->GetBufferSize());
               ++n;
            }
         }
         if (fWriteBasket >= fMaxBaskets) {
            //old versions may need this fix
            ExpandBasketArrays();
            fBasketBytes[fWriteBasket] = fBasketBytes[fWriteBasket-1];
            fBasketEntry[fWriteBasket] = fEntries;
            fBasketSeek [fWriteBasket] = fBasketSeek [fWriteBasket-1];

         }
         if (!fSplitLevel && fBranches.GetEntriesFast()) fSplitLevel = 1;
         gROOT->SetReadingObject(kFALSE);
         if (IsA() == TBranch::Class()) {
            if (fNleaves == 0) {
               fReadLeaves = &TBranch::ReadLeaves0Impl;
            } else if (fNleaves == 1) {
               fReadLeaves = &TBranch::ReadLeaves1Impl;
            } else if (fNleaves == 2) {
               fReadLeaves = &TBranch::ReadLeaves2Impl;
            } else {
               fReadLeaves = &TBranch::ReadLeavesImpl;
            }
         }
         return;
      }
      //====process old versions before automatic schema evolution
      Int_t n,i,j,ijunk;
      if (v > 5) {
         Stat_t djunk;
         TNamed::Streamer(b);
         if (v > 7) TAttFill::Streamer(b);
         b >> fCompress;
         b >> fBasketSize;
         b >> fEntryOffsetLen;
         b >> fWriteBasket;
         b >> ijunk; fEntryNumber = (Long64_t)ijunk;
         b >> fOffset;
         b >> fMaxBaskets;
         if (v > 6) b >> fSplitLevel;
         b >> djunk; fEntries  = (Long64_t)djunk;
         b >> djunk; fTotBytes = (Long64_t)djunk;
         b >> djunk; fZipBytes = (Long64_t)djunk;

         fBranches.Streamer(b);
         fLeaves.Streamer(b);
         fBaskets.Streamer(b);
         fBasketBytes = new Int_t[fMaxBaskets];
         fBasketEntry = new Long64_t[fMaxBaskets];
         fBasketSeek  = new Long64_t[fMaxBaskets];
         Char_t isArray;
         b >> isArray;
         b.ReadFastArray(fBasketBytes,fMaxBaskets);
         b >> isArray;
         for (i=0;i<fMaxBaskets;i++) {b >> ijunk; fBasketEntry[i] = ijunk;}
         b >> isArray;
         for (i=0;i<fMaxBaskets;i++) {
            if (isArray == 2) b >> fBasketSeek[i];
            else              {Int_t bsize; b >> bsize; fBasketSeek[i] = (Long64_t)bsize;};
         }
         fFileName.Streamer(b);
         b.CheckByteCount(R__s, R__c, TBranch::IsA());
         fDirectory = 0;
         fNleaves = fLeaves.GetEntriesFast();
         for (i=0;i<fNleaves;i++) {
            TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(i);
            leaf->SetBranch(this);
         }
         fNBaskets = fBaskets.GetEntries();
         for (j=fWriteBasket,n=0;j>=0 && n<fNBaskets;--j) {
            TBasket *bk = (TBasket*)fBaskets.UncheckedAt(j);
            if (bk) {
               bk->SetBranch(this);
               GetTree()->IncrementTotalBuffers(bk->GetBufferSize());
               ++n;
            }
         }
         if (fWriteBasket >= fMaxBaskets) {
            //old versions may need this fix
            ExpandBasketArrays();
            fBasketBytes[fWriteBasket] = fBasketBytes[fWriteBasket-1];
            fBasketEntry[fWriteBasket] = fEntries;
            fBasketSeek [fWriteBasket] = fBasketSeek [fWriteBasket-1];

         }
         // Check Byte Count is not needed since it was done in ReadBuffer
         if (!fSplitLevel && fBranches.GetEntriesFast()) fSplitLevel = 1;
         gROOT->SetReadingObject(kFALSE);
         b.CheckByteCount(R__s, R__c, TBranch::IsA());
         if (IsA() == TBranch::Class()) {
            if (fNleaves == 0) {
               fReadLeaves = &TBranch::ReadLeaves0Impl;
            } else if (fNleaves == 1) {
               fReadLeaves = &TBranch::ReadLeaves1Impl;
            } else if (fNleaves == 2) {
               fReadLeaves = &TBranch::ReadLeaves2Impl;
            } else {
               fReadLeaves = &TBranch::ReadLeavesImpl;
            }
         }
         return;
      }
      //====process very old versions
      Stat_t djunk;
      TNamed::Streamer(b);
      b >> fCompress;
      b >> fBasketSize;
      b >> fEntryOffsetLen;
      b >> fMaxBaskets;
      b >> fWriteBasket;
      b >> ijunk; fEntryNumber = (Long64_t)ijunk;
      b >> djunk; fEntries  = (Long64_t)djunk;
      b >> djunk; fTotBytes = (Long64_t)djunk;
      b >> djunk; fZipBytes = (Long64_t)djunk;
      b >> fOffset;
      fBranches.Streamer(b);
      fLeaves.Streamer(b);
      fNleaves = fLeaves.GetEntriesFast();
      for (i=0;i<fNleaves;i++) {
         TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(i);
         leaf->SetBranch(this);
      }
      fBaskets.Streamer(b);
      Int_t nbaskets = fBaskets.GetEntries();
      for (j=fWriteBasket,n=0;j>0 && n<nbaskets;--j) {
         TBasket *bk = (TBasket*)fBaskets.UncheckedAt(j);
         if (bk) {
            bk->SetBranch(this);
            GetTree()->IncrementTotalBuffers(bk->GetBufferSize());
            ++n;
         }
      }
      fBasketEntry = new Long64_t[fMaxBaskets];
      b >> n;
      for (i=0;i<n;i++) {b >> ijunk; fBasketEntry[i] = ijunk;}
      fBasketBytes = new Int_t[fMaxBaskets];
      if (v > 4) {
         n  = b.ReadArray(fBasketBytes);
      } else {
         for (n=0;n<fMaxBaskets;n++) fBasketBytes[n] = 0;
      }
      if (v < 2) {
         fBasketSeek = new Long64_t[fMaxBaskets];
         for (n=0;n<fWriteBasket;n++) {
            fBasketSeek[n] = GetBasket(n)->GetSeekKey();
         }
      } else {
         fBasketSeek = new Long64_t[fMaxBaskets];
         b >> n;
         for (n=0;n<fMaxBaskets;n++) {
            Int_t aseek;
            b >> aseek;
            fBasketSeek[n] = Long64_t(aseek);
         }
      }
      if (v > 2) {
         fFileName.Streamer(b);
      }
      fDirectory = 0;
      if (v < 4) SetAutoDelete(kTRUE);
      if (!fSplitLevel && fBranches.GetEntriesFast()) fSplitLevel = 1;
      gROOT->SetReadingObject(kFALSE);
      b.CheckByteCount(R__s, R__c, TBranch::IsA());
      //====end of old versions
      if (IsA() == TBranch::Class()) {
         if (fNleaves == 0) {
            fReadLeaves = &TBranch::ReadLeaves0Impl;
         } else if (fNleaves == 1) {
            fReadLeaves = &TBranch::ReadLeaves1Impl;
         } else if (fNleaves == 2) {
            fReadLeaves = &TBranch::ReadLeaves2Impl;
         } else {
            fReadLeaves = &TBranch::ReadLeavesImpl;
         }
      }
   } else {
      Int_t maxBaskets = fMaxBaskets;
      fMaxBaskets = fWriteBasket+1;
      if (fMaxBaskets < 10) fMaxBaskets=10;
      TBasket *writebasket = 0;
      if (fNBaskets == 1) {
         writebasket = (TBasket*)fBaskets.UncheckedAt(fWriteBasket);
         if (writebasket && writebasket->GetNevBuf()==0) {
            fBaskets[fWriteBasket] = 0;
         } else {
            writebasket = 0;
         }
      }
      b.WriteClassBuffer(TBranch::Class(),this);
      if (writebasket) {
         fBaskets[fWriteBasket] = writebasket;         
      }
      fMaxBaskets = maxBaskets;
   }
}

//_______________________________________________________________________
Int_t TBranch::WriteBasket(TBasket* basket, Int_t where)
{
   // Write the current basket to disk and return the number of bytes
   // written to the file.

   Int_t nevbuf = basket->GetNevBuf();
   if (fEntryOffsetLen > 10 &&  (4*nevbuf) < fEntryOffsetLen ) {
      // Make sure that the fEntryOffset array does not stay large unnecessarily.
      fEntryOffsetLen = nevbuf < 3 ? 10 : 4*nevbuf; // assume some fluctuations.
   } else if (fEntryOffsetLen && nevbuf > fEntryOffsetLen) {
      // Increase the array ... 
      fEntryOffsetLen = 2*nevbuf; // assume some fluctuations.
   }

   Int_t nout  = basket->WriteBuffer();    //  Write buffer
   fBasketBytes[where]  = basket->GetNbytes();
   fBasketSeek[where]   = basket->GetSeekKey();
   Int_t addbytes = basket->GetObjlen() + basket->GetKeylen();
   TBasket *reusebasket = 0;
   if (nout>0) {
      // The Basket was written so we can now safely reuse it.
      fBaskets[where] = 0;

      reusebasket = basket;
      reusebasket->Reset();
   }
   fZipBytes += nout;
   fTotBytes += addbytes;
   fTree->AddTotBytes(addbytes);
   fTree->AddZipBytes(nout);

   if (where==fWriteBasket) {
      ++fWriteBasket;
      if (fWriteBasket >= fMaxBaskets) {
         ExpandBasketArrays();
      }
      fBaskets.AddAtAndExpand(reusebasket,fWriteBasket);
      fBasketEntry[fWriteBasket] = fEntryNumber;
   } else {
      --fNBaskets;
      fBaskets[where] = 0;
      basket->DropBuffers();
      if (basket == fCurrentBasket) {
         fCurrentBasket    = 0;
         fFirstBasketEntry = -1;
         fNextBasketEntry  = -1;
      }      
      delete basket;
   }

   return nout;
}

//------------------------------------------------------------------------------
void TBranch::SetFirstEntry(Long64_t entry)
{
   //set the first entry number (case of TBranchSTL)
   fFirstEntry = entry;
   fEntries = 0;
   fEntryNumber = entry;
   if( fBasketEntry )
      fBasketEntry[0] = entry;
   for( Int_t i = 0; i < fBranches.GetEntriesFast(); ++i )
      ((TBranch*)fBranches[i])->SetFirstEntry( entry );
}

//______________________________________________________________________________
void TBranch::SetupAddresses()
{
   // -- If the branch address is not set,  we set all addresses starting with
   // the top level parent branch.  
   
   // Nothing to do for regular branch, the TLeaf already did it.
}

//______________________________________________________________________________
void TBranch::UpdateFile()
{
   // Refresh the value of fDirectory (i.e. where this branch writes/reads its buffers)
   // with the current value of fTree->GetCurrentFile unless this branch has been
   // redirected to a different file.  Also update the sub-branches.

   TFile *file = fTree->GetCurrentFile();
   if (fFileName.Length() == 0) {
      fDirectory = file;

      // Apply to all existing baskets.
      TIter nextb(GetListOfBaskets());
      TBasket *basket;
      while ((basket = (TBasket*)nextb())) {
         basket->SetParent(file);
      }
   }

   // Apply to sub-branches as well.
   TIter next(GetListOfBranches());
   TBranch *branch;
   while ((branch = (TBranch*)next())) {
      branch->UpdateFile();
   }
}
