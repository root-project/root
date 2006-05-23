// @(#)root/tree:$Name:  $:$Id: TBranchElement.cxx,v 1.195 2006/05/15 11:01:14 rdm Exp $
// Authors Rene Brun , Philippe Canal, Markus Frank  14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchElement                                                       //
//                                                                      //
// A Branch for the case of an object                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TFile.h"
#include "TBranchElement.h"
#include "TBranchObject.h"
#include "TBranchRef.h"
#include "TClonesArray.h"
#include "TTree.h"
#include "TBasket.h"
#include "TLeafElement.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TDataType.h"
#include "TDataMember.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TBrowser.h"
#include "TFolder.h"
#include "TRealData.h"
#include "Api.h"
#include "TError.h"
#include "TVirtualCollectionProxy.h"

const Int_t kMaxLen = 1024;
R__EXTERN  TTree *gTree;

ClassImp(TBranchElement)

//______________________________________________________________________________
TBranchElement::TBranchElement(): TBranch(), fCurrentClass(), fParentClass(), fBranchClass()
{
//*-*-*-*-*-*Default constructor for BranchElement*-*-*-*-*-*-*-*-*-*
//*-*        ====================================

   fNleaves       = 1;
   fInfo          = 0;
   fBranchCount   = 0;
   fBranchCount2  = 0;
   fObject        = 0;
   fMaximum       = 0;
   fBranchPointer = 0;
   fNdata         = 1;
   fSTLtype       = TClassEdit::kNotSTL;
   fCollProxy     = 0;
   fCheckSum      = 0;
   fBranchOffset  = 0;
   fBranchTypes   = 0;
   fInit = fInitOffsets = kFALSE;
   fType          = 0;
}


//______________________________________________________________________________
TBranchElement::TBranchElement(const char *bname, TStreamerInfo *sinfo, Int_t id, char *pointer, Int_t basketsize, Int_t splitlevel, Int_t btype)
    :TBranch(), fCurrentClass(), fParentClass(), fBranchClass(sinfo->GetClass())
{
// Create a BranchElement
//
// If splitlevel > 0 this branch in turn is split into sub branches

   fCollProxy = 0;
   if (gDebug > 0) {
      printf("BranchElement, bname=%s, sinfo=%s, id=%d, splitlevel=%d\n",bname,sinfo->GetName(),id,splitlevel);
   }
   char name[kMaxLen];
   strcpy(name,bname);

   SetName(name);
   SetTitle(name);

   fSplitLevel   = splitlevel;
   if (id < 0)     splitlevel = 0;

   TClass *cl    = sinfo->GetClass();
   fInfo         = sinfo;
   fCheckSum     = sinfo->GetCheckSum();
   fID           = id;
   fInit         = kTRUE;
   fStreamerType = -1;
   fType         = 0;
   fBranchCount  = 0;
   fBranchCount2 = 0;
   fObject       = 0;
   fBranchPointer= 0;
   fNdata        = 1;
   fSTLtype      = TClassEdit::kNotSTL;
   fClassVersion = cl->GetClassVersion();
   fTree         = gTree;
   fMaximum      = 0;
   fBranchOffset = 0;
   fBranchTypes  = 0;
   fInitOffsets  = kFALSE;
   ULong_t *elems = sinfo->GetElems();
   TStreamerElement *element = 0;
   TBranchElement *brcount = 0;
   if (id >= 0) {
      element = (TStreamerElement *)elems[id];
      fStreamerType = element->GetType();
      if (fStreamerType == TStreamerInfo::kObject
         || fStreamerType == TStreamerInfo::kBase
         || fStreamerType == TStreamerInfo::kTNamed
         || fStreamerType == TStreamerInfo::kTObject
         || fStreamerType == TStreamerInfo::kObjectp
         || fStreamerType == TStreamerInfo::kObjectP) {
         if (cl->InheritsFrom(TObject::Class())) SetBit(kBranchObject);
      }
      if (element->IsA() == TStreamerBasicPointer::Class()) {
         TStreamerBasicPointer *bp = (TStreamerBasicPointer *)element;
         char countname[kMaxLen];
         strcpy(countname,bname);
         char *dot = strrchr(countname,'.');
         if (dot) *(dot+1) = 0;
         else countname[0] = 0;
         strcat(countname,bp->GetCountName());
         brcount = (TBranchElement *)fTree->GetBranch(countname);
         sprintf(countname,"%s[%s]",name,bp->GetCountName());
         SetTitle(countname);
      }
   } else {
      if (cl->InheritsFrom(TObject::Class())) SetBit(kBranchObject);
   }

   // Set the bit kAutoDelete to specify that when reading
   // the object should be deleted before calling Streamer.
   // It is foreseen to not set this bit in a future version.
   //SetAutoDelete(kTRUE);
   SetAutoDelete(kFALSE);

   fDirectory   = fTree->GetDirectory();
   fFileName    = "";
   fClassName   = sinfo->GetName();
   if (gDebug > 1) printf("Building Branch=%s, class=%s, info=%s, version=%d, id=%d, fStreamerType=%d, btype=%d\n",bname,cl->GetName(),sinfo->GetName(),fClassVersion,id,fStreamerType,btype);
   fCompress = -1;
   if (gTree->GetDirectory()) {
      TFile *bfile = gTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }
   //change defaults set in TBranch constructor
   fEntryOffsetLen = 0;
   if (btype || fStreamerType <= 0
             || fStreamerType == 7
             || fStreamerType > 15) fEntryOffsetLen = 1000;
   if (basketsize < 100+fEntryOffsetLen) basketsize = 100+fEntryOffsetLen;
   fBasketSize     = basketsize;
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketEntry    = new Long64_t[fMaxBaskets];
   fBasketSeek     = new Long64_t[fMaxBaskets];

   for (Int_t i=0;i<fMaxBaskets;i++) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i]  = 0;
   }

   // The fBits part of a TObject is of varying length,
   // so we must ask the TBranch to inform the TBasket
   // that we need a fEntryOffset table created.
   //
   // Note: The fBits is varying size because the pidf
   //       is streamed only when the TObject is referenced
   //       by a TRef.

   if (fStreamerType == TStreamerInfo::kBits) {
      fEntryOffsetLen = 1000;
   }

   // Create a basket for the terminal branch
   TBasket *basket = new TBasket(name,fTree->GetName(),this);
   fBaskets.Add(basket);

   // save pointer (if non null). Will be used in Unroll in case we find
   // a TClonesArray in a derived class.
   if (pointer) fBranchPointer = pointer;

   // create sub branches if requested by splitlevel

   if (splitlevel > 0) {
      const char* elem_type = element->GetTypeName();
      fSTLtype = TMath::Abs(TClassEdit::IsSTLCont(elem_type));
      TClass *clm;
      if (element->CannotSplit()) {
         //printf("element: %s/%s will not be split\n",element->GetName(),element->GetTitle());
         fSplitLevel = 0;
      } else if (element->IsA() == TStreamerBase::Class()) {
         // ===> develop the base class
         fType = 1;
         clm = gROOT->GetClass(element->GetName());
         Int_t nbranches = fBranches.GetEntriesFast();
         if (!strcmp(name,clm->GetName())) Unroll("",cl,clm,basketsize,splitlevel,0);
         else                              Unroll(name,clm,clm,basketsize,splitlevel,0);
         if (!strcmp(name,clm->GetName())) return;
         if (strchr(bname,'.')) return;
         if (nbranches == fBranches.GetEntriesFast()) {
            if (strlen(bname)) sprintf(name,"%s.%s",bname,clm->GetName());
            else               sprintf(name,"%s",clm->GetName());
            SetName(name);
            SetTitle(name);
         }
         return;

      } else if (!strcmp(elem_type,"TClonesArray") || !strcmp(elem_type,"TClonesArray*")) {
         Bool_t ispointer = !strcmp(elem_type,"TClonesArray*");
         TClonesArray *clones;
         if (ispointer) {
            char **ppointer = (char**)(pointer);
            clones = (TClonesArray*)(*ppointer);
         } else {
            clones = (TClonesArray*)pointer;
         }
         basket->DeleteEntryOffset(); //entryoffset not required for the clonesarray counter
         fEntryOffsetLen = 0;

         // ===> Create a leafcount
         TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
         leaf->SetBranch(this);
         fNleaves = 1;
         fLeaves.Add(leaf);
         fTree->GetListOfLeaves()->Add(leaf);

         if (!clones) return;
         clm = clones->GetClass();
         if (!clm) return;

         // Create a basket for the leafcount
         TBasket *basket2 = new TBasket(name,fTree->GetName(),this);
         fBaskets.Add(basket2);

         // ===> create sub branches for each data member of a TClonesArray
         fType = 3;
         //check that the contained objects class name is part of the element title
         //This name is mandatory when reading the Tree later on and
         //the parent class with the pointer to the TClonesArray is not available.
         //This info will be used by TStreamerInfo::New
         fClonesName = clm->GetName();
         char aname[100];
         sprintf(aname," (%s)",clm->GetName());
         TString atitle = element->GetTitle();
         if (!atitle.Contains(aname)) {
            atitle += aname;
            element->SetTitle(atitle.Data());
         }
         char branchname[kMaxLen];
         sprintf(branchname,"%s_",name);
         SetTitle(branchname);
         leaf->SetName(branchname);
         leaf->SetTitle(branchname);
         Unroll(name,clm,clm,basketsize,splitlevel,31);
         BuildTitle(name);
         return;

      } else if ( (fSTLtype>= TClassEdit::kVector   && fSTLtype<= TClassEdit::kMultiSet) ||
                  (fSTLtype>=-TClassEdit::kMultiSet && fSTLtype<=-TClassEdit::kVector) ) {
         // case of vector<>, list<> deque<>, set<>, multiset<>
         TClass *contCl = gROOT->GetClass(elem_type);
         fCollProxy = contCl->GetCollectionProxy()->Generate();
         clm = fCollProxy->GetValueClass();

         // See if we can split:
         Bool_t cansplit = kTRUE;
         if (clm==0) {
            cansplit = kFALSE;
         } else if (clm==TString::Class() || clm==gROOT->GetClass("string")) {
            cansplit = kFALSE;
         } else if (fCollProxy->HasPointers()) {
            cansplit = kFALSE;
         } else if ( !clm->CanSplit() ) {
            cansplit = kFALSE;
         } else if ( clm->GetCollectionProxy() != 0 ) {
            // A collection was stored in a collection, we do not know how to split it.
            cansplit = kFALSE;
         }

         // ===> Create a leafcount
         if (cansplit) {

            TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
            leaf->SetBranch(this);
            fNleaves = 1;
            fLeaves.Add(leaf);
            fTree->GetListOfLeaves()->Add(leaf);
            // Create a basket for the leafcount
            TBasket *basket2 = new TBasket(name,fTree->GetName(),this);
            fBaskets.Add(basket2);

            // ===> create sub branches for each data member of a TClonesArray
            fType = 4;

            //check that the contained objects class name is part of the element title
            //This name is mandatory when reading the Tree later on and
            //the parent class with the pointer to the TClonesArray is not available.
            //This info will be used by TStreamerInfo::New
            fClonesName = clm->GetName();
            char aname[100];
            sprintf(aname," (%s)",clm->GetName());
            TString atitle = element->GetTitle();
            if (!atitle.Contains(aname)) {
               atitle += aname;
               element->SetTitle(atitle.Data());
            }
            char branchname[kMaxLen];
            sprintf(branchname,"%s_",name);
            SetTitle(branchname);
            leaf->SetName(branchname);
            leaf->SetTitle(branchname);
            Unroll(name,clm,clm,basketsize,splitlevel,41);
            BuildTitle(name);
            return;
         }

      } else if (!strchr(elem_type,'*') &&
                 (fStreamerType == TStreamerInfo::kObject || fStreamerType == TStreamerInfo::kAny)) {
         // ===> create sub branches for members that are classes
         fType = 2;
         clm = gROOT->GetClass(elem_type);
         if (Unroll(name,clm,clm,basketsize,splitlevel,0) >= 0) return;

      }
   }

   TLeaf *leaf = new TLeafElement(GetTitle(),fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   leaf->SetBranch(this);
   fNleaves = 1;
   fLeaves.Add(leaf);
   fTree->GetListOfLeaves()->Add(leaf);
   if (brcount) SetBranchCount(brcount);
}

//______________________________________________________________________________
TBranchElement::TBranchElement(const char *bname, TClonesArray *clones, Int_t basketsize, Int_t splitlevel, Int_t compress)
    :TBranch(), fInfo(TClonesArray::Class()->GetStreamerInfo()),
     fCurrentClass(),fParentClass(), fBranchClass(fInfo->GetClass())
{
// Create a BranchElement
//
// If splitlevel > 0 this branch in turn is split into sub branches

   fCollProxy = 0;
   fSplitLevel    = splitlevel;
   fID            = 0;
   fInit          = kTRUE;
   fStreamerType  = -1;
   fType          = 0;
   fClassVersion  = TClonesArray::Class()->GetClassVersion();
   fCheckSum      = fInfo->GetCheckSum();
   fBranchCount   = 0;
   fBranchCount2  = 0;
   fObject        = 0;
   fBranchPointer = 0;
   fMaximum       = 0;
   fBranchOffset  = 0;
   fBranchTypes   = 0;
   fSTLtype       = TClassEdit::kNotSTL;
   fInitOffsets   = kFALSE;

   fTree          = gTree;
   fDirectory     = fTree->GetDirectory();
   fFileName      = "";

   SetName(bname);
   const char* name = GetName();
   SetTitle(name);
   fClassName = fInfo->GetName();
   fCompress = compress;
   if (compress == -1 && fTree->GetDirectory()) {
      TFile *bfile = fTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }

   if (basketsize < 100) basketsize = 100;
   fBasketSize     = basketsize;
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketEntry    = new Long64_t[fMaxBaskets];
   fBasketSeek     = new Long64_t[fMaxBaskets];

   for (Int_t i=0;i<fMaxBaskets;i++) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i]  = 0;
   }

   // Create a basket for the terminal branch
   TBasket *basket = new TBasket(name,fTree->GetName(),this);
   fBaskets.Add(basket);

   // Set the bit kAutoDelete to specify that when reading
   // the object should be deleted before calling Streamer.
   // It is foreseen to not set this bit in a future version.
   //SetAutoDelete(kTRUE);
   SetAutoDelete(kFALSE);


   // create sub branches if requested by splitlevel
   if (splitlevel > 0) {
      // ===> Create a leafcount
      TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
      leaf->SetBranch(this);
      fNleaves = 1;
      fLeaves.Add(leaf);
      fTree->GetListOfLeaves()->Add(leaf);
      // Create a basket for the leafcount
      TBasket *basket = new TBasket(name,fTree->GetName(),this);
      fBaskets.Add(basket);
      // ===> create sub branches for each data member of a TClonesArray
      fType = 3;
      TClass *clm = clones->GetClass();
      if (!clm) return;
      fClonesName = clm->GetName();
      std::string branchname = name + std::string("_");
      SetTitle(branchname.c_str());
      leaf->SetName(branchname.c_str());
      leaf->SetTitle(branchname.c_str());
      Unroll(name,clm,clm,basketsize,splitlevel,31);
      BuildTitle(name);
      return;
   }

   SetBit(kBranchObject);

   TLeaf *leaf     = new TLeafElement(GetTitle(),fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   leaf->SetBranch(this);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);
}
//______________________________________________________________________________
TBranchElement::TBranchElement(const char *bname, TVirtualCollectionProxy *cont, Int_t basketsize, Int_t splitlevel, Int_t compress)
  : TBranch(),fCurrentClass(),fParentClass(), fBranchClass(cont->GetCollectionClass())
{
   // Create a BranchElement
   //
   // If splitlevel > 0 this branch in turn is split into sub branches

   fCollProxy = cont->Generate();
   char name[kMaxLen];
   strcpy(name,bname);
   if (name[strlen(name)-1]=='.') name[strlen(name)-1]=0;
   fInitOffsets   = kFALSE;
   fSplitLevel    = splitlevel;
   fInfo          = 0;
   fID            = -1;
   fInit          = kTRUE;
   fStreamerType  = -1; // TStreamerInfo::kSTLp;
   fType          = 0;
   fClassVersion  = cont->GetCollectionClass()->GetClassVersion();
   fBranchCount   = 0;
   fBranchCount2  = 0;
   fObject        = 0;
   fBranchPointer = 0;
   fMaximum       = 0;
   fBranchOffset  = 0;
   fBranchTypes   = 0;
   fSTLtype       = TClassEdit::kNotSTL;

   fTree          = gTree;
   fDirectory     = fTree->GetDirectory();
   fFileName      = "";

   SetName(name);
   SetTitle(name);
   //Bool_t Implemented=kFALSE;
   //R__ASSERT(Implemented);
   fClassName    = fBranchClass->GetName();
   fCompress = compress;
   if (compress == -1 && fTree->GetDirectory()) {
      TFile *bfile = fTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }

   if (basketsize < 100) basketsize = 100;
   fBasketSize     = basketsize;
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketEntry    = new Long64_t[fMaxBaskets];
   fBasketSeek     = new Long64_t[fMaxBaskets];

   fBasketEntry[0] = fEntryNumber;
   fBasketBytes[0] = 0;

   // Create a basket for the terminal branch
   TBasket *basket = new TBasket(name,fTree->GetName(),this);
   fBaskets.Add(basket);

   // Set the bit kAutoDelete to specify that when reading
   // the object should be deleted before calling Streamer.
   // It is foreseen to not set this bit in a future version.
   //SetAutoDelete(kTRUE);
   SetAutoDelete(kFALSE);


   // create sub branches if requested by splitlevel
   if (splitlevel > 0 && fBranchClass->CanSplit()) {
      // ===> Create a leafcount
      TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
      leaf->SetBranch(this);
      fNleaves = 1;
      fLeaves.Add(leaf);
      fTree->GetListOfLeaves()->Add(leaf);
      // Create a basket for the leafcount
      TBasket *basket = new TBasket(name,fTree->GetName(),this);
      fBaskets.Add(basket);
      // ===> create sub branches for each data member of a TSTLCont
      fType = 4;
      TClass *clm = cont->GetValueClass();
      if (!clm) return;
      fClonesName = clm->GetName();
      char branchname[kMaxLen];
      sprintf(branchname,"%s_",name);
      SetTitle(branchname);
      leaf->SetName(branchname);
      leaf->SetTitle(branchname);
      Unroll(name,clm,clm,basketsize,splitlevel,41);
      BuildTitle(name);
      return;
   }

   TLeaf *leaf     = new TLeafElement(GetTitle(),fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   leaf->SetBranch(this);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);
}

//______________________________________________________________________________
TBranchElement::TBranchElement(const TBranchElement& tbe) :
  TBranch(tbe),
  fClassName(tbe.fClassName),
  fParentName(tbe.fParentName),
  fClonesName(tbe.fClonesName),
  fCollProxy(tbe.fCollProxy),
  fCheckSum(tbe.fCheckSum),
  fClassVersion(tbe.fClassVersion),
  fID(tbe.fID),
  fType(tbe.fType),
  fStreamerType(tbe.fStreamerType),
  fMaximum(tbe.fMaximum),
  fSTLtype(tbe.fSTLtype),
  fNdata(tbe.fNdata),
  fBranchCount(tbe.fBranchCount),
  fBranchCount2(tbe.fBranchCount2),
  fInfo(tbe.fInfo),
  fObject(tbe.fObject),
  fBranchPointer(tbe.fBranchPointer),
  fInit(tbe.fInit),
  fInitOffsets(tbe.fInitOffsets),
  fCurrentClass(tbe.fCurrentClass),
  fParentClass(tbe.fParentClass),
  fBranchClass(tbe.fBranchClass),
  fBranchOffset(tbe.fBranchOffset),
  fBranchTypes(tbe.fBranchTypes)
{ }

//______________________________________________________________________________
TBranchElement& TBranchElement::operator=(const TBranchElement& tbe)
{
  if(this!=&tbe) {
    TBranch::operator=(tbe);
    fClassName=tbe.fClassName;
    fParentName=tbe.fParentName;
    fClonesName=tbe.fClonesName;
    fCollProxy=tbe.fCollProxy;
    fCheckSum=tbe.fCheckSum;
    fClassVersion=tbe.fClassVersion;
    fID=tbe.fID;
    fType=tbe.fType;
    fStreamerType=tbe.fStreamerType;
    fMaximum=tbe.fMaximum;
    fSTLtype=tbe.fSTLtype;
    fNdata=tbe.fNdata;
    fBranchCount=tbe.fBranchCount;
    fBranchCount2=tbe.fBranchCount2;
    fInfo=tbe.fInfo;
    fObject=tbe.fObject;
    fBranchPointer=tbe.fBranchPointer;
    fInit=tbe.fInit;
    fInitOffsets=tbe.fInitOffsets;
    fCurrentClass=tbe.fCurrentClass;
    fParentClass=tbe.fParentClass;
    fBranchClass=tbe.fBranchClass;
    fBranchOffset=tbe.fBranchOffset;
    fBranchTypes=tbe.fBranchTypes;
  } return *this; 
}

//______________________________________________________________________________
TBranchElement::~TBranchElement()
{
//*-*-*-*-*-*Default destructor for a BranchElement*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =====================================

   if ( fType==4      ) delete fCollProxy;
   if ( fBranchOffset ) delete [] fBranchOffset;
   if ( fBranchTypes  ) delete [] fBranchTypes;

   fCollProxy=0;
   fBranches.Delete();

   //SetAddress may have allocated an object. Must delete it
   if (TestBit(kDeleteObject)) {
      if (fObject) {
         //TObject *obj = (TObject*)fObject;
         // objects of emulated classes allocated in SetAddress should be deleted
         //printf("deleting fObject=%x of class: %s, branch: %s, fAddress=%x\n",obj,fClassName.Data(),GetName(),fAddress);
         //delete obj;
      }
   }
}

//______________________________________________________________________________
TBranch *TBranchElement::Branch(const char *subname, void *address, const char *leaflist,Int_t bufsize)
{
//  Create a sub-branch of this branch (simple variable case)
//  ==================================
//
//     This Branch constructor is provided to support non-objects in
//     a Tree. The variables described in leaflist may be simple variables
//     or structures.
//    See the two following constructors for writing objects in a Tree.
//
//    By default the branch buffers are stored in the same file as the Tree.
//    use TBranch::SetFile to specify a different file

   gTree = fTree;
   std::string name(GetName()+std::string(".")+subname);
   TBranch *branch = new TBranch(name.c_str(),address,leaflist,bufsize);
   if (branch->IsZombie()) {
      delete branch;
      return 0;
   }
   TLeaf *leaf;
   TIter next(branch->GetListOfLeaves());
   while ((leaf = (TLeaf*)next())) {
      std::string name(GetName()+std::string(".")+leaf->GetName());
      leaf->SetName(name.c_str());
   }
   fBranches.Add(branch);
   return branch;
}


//______________________________________________________________________________
Int_t TBranchElement::Branch(const char *foldername, Int_t bufsize, Int_t splitlevel)
{
//   This function creates one sub-branch for each element in the folder.
//   The function returns the total number of branches created.

   TObject *ob = gROOT->FindObjectAny(foldername);
   if (!ob) return 0;
   if (ob->IsA() != TFolder::Class()) return 0;
   Int_t nbranches = GetListOfBranches()->GetEntries();
   TFolder *folder = (TFolder*)ob;
   TIter next(folder->GetListOfFolders());
   TObject *obj;
   char occur[20];
   while ((obj=next())) {
      std::string curname(foldername+std::string("/")+obj->GetName());
      if (obj->IsA() == TFolder::Class()) {
         Branch(curname.c_str(), bufsize, splitlevel-1);
      } else {
         void *add = (void*)folder->GetListOfFolders()->GetObjectRef(obj);
         for (size_t i=curname.find('/'); i != std::string::npos; i=curname.find('/')) {
            curname[i] = '.';
         }
         Int_t noccur = folder->Occurence(obj);
         if (noccur > 0) {
            sprintf(occur,"_%d",noccur);
            curname += occur;
         }
         TBranchElement *br;
         br = (TBranchElement*)Branch(curname.c_str(),obj->ClassName(), add, bufsize, splitlevel-1);
         br->SetBranchFolder();
      }
   }
   return GetListOfBranches()->GetEntries() - nbranches;
}


//______________________________________________________________________________
TBranch *TBranchElement::Branch(const char *subname, const char *classname, void *add, Int_t bufsize, Int_t splitlevel)
{
//  Create a sub-branch of this branch (with a class)
//  ==================================
//
//    Build a TBranchElement for an object of class classname.
//    addobj is the address of a pointer to an object of class classname.
//    The class dictionary must be available (ClassDef in class header).
//
//    This option requires access to the library where the corresponding class
//    is defined. Accessing one single data member in the object implies
//    reading the full object.
//
//    By default the branch buffers are stored in the same file as the parent branch.
//    use TBranch::SetFile to specify a different file
//
//    see IMPORTANT NOTE about branch names in TTree::Bronch
//
//   Use splitlevel < 0 instead of splitlevel=0 when the class
//   has a custom Streamer

   gTree = fTree;
   TClass *cl = gROOT->GetClass(classname);
   if (!cl) {
      Error("Branch","Cannot find class:%s",classname);
      return 0;
   }

   //if splitlevel <= 0 and class has a custom Streamer, we must create
   //a TBranchObject. We cannot assume that TClass::ReadBuffer is consistent
   //with the custom Streamer. The penalty is that one cannot process
   //this Tree without the class library containing the class.
   //The following convention is used for the RootFlag
   // #pragma link C++ class TExMap;     rootflag = 0
   // #pragma link C++ class TList-;     rootflag = 1
   // #pragma link C++ class TArray!;    rootflag = 2
   // #pragma link C++ class TArrayC-!;  rootflag = 3
   // #pragma link C++ class TBits+;     rootflag = 4
   // #pragma link C++ class Txxxx+!;    rootflag = 6

   char **ppointer = (char**)add;
   char *objadd = *ppointer;
   std::string name(GetName()+std::string(".")+subname);
   if (cl == TClonesArray::Class()) {
      TClonesArray *clones = (TClonesArray *)objadd;
      if (!clones) {
         Error("Branch","Pointer to TClonesArray is null");
         return 0;
      }
      if (!clones->GetClass()) {
         Error("Branch","TClonesArray with no class defined in branch: %s",name.c_str());
         return 0;
      }
      G__ClassInfo *classinfo = clones->GetClass()->GetClassInfo();
      if (!classinfo) {
         Error("Bronch","TClonesArray with no dictionary defined in branch: %s",name.c_str());
         return 0;
      }
      if (splitlevel > 0) {
         if (classinfo->RootFlag() & 1)
            Warning("Branch","Using split mode on a class: %s with a custom Streamer",clones->GetClass()->GetName());
      } else {
         if (classinfo->RootFlag() & 1) clones->BypassStreamer(kFALSE);
         TBranchObject *branch = new TBranchObject(name.c_str(),classname,add,bufsize,0);
         fBranches.Add(branch);
         return branch;
      }
   }

   Bool_t hasCustomStreamer = kFALSE;
   if (!cl->GetClassInfo()) {
      Error("Branch","Cannot find dictionary for class: %s",classname);
      return 0;
   }
   if (cl->GetClassInfo()->RootFlag() & 1)  hasCustomStreamer = kTRUE;
   if (splitlevel < 0 || (splitlevel == 0 && hasCustomStreamer)) {
      TBranchObject *branch = new TBranchObject(name.c_str(),classname,add,bufsize,0);
      fBranches.Add(branch);
      return branch;
   }

   //hopefully normal case
   Bool_t delobj = kFALSE;
   //====> special case of TClonesArray
   if(cl == TClonesArray::Class()) {
      TBranchElement *branch = new TBranchElement(name.c_str(),(TClonesArray*)objadd,bufsize,splitlevel);
      fBranches.Add(branch);
      branch->SetAddress(add);
      return branch;
   }
   //====>

   if (!objadd) {
      objadd = (char*)cl->New();
      *ppointer = objadd;
      delobj = kTRUE;
   }
   //build the StreamerInfo if first time for the class
   Bool_t optim = TStreamerInfo::CanOptimize();
   if (splitlevel > 0) TStreamerInfo::Optimize(kFALSE);
   TStreamerInfo *sinfo = fTree->BuildStreamerInfo(cl,objadd);
   TStreamerInfo::Optimize(optim);

   // create a dummy top level  branch object
   Int_t id = -1;
   if (splitlevel > 0) id = -2;
   size_t dot = name.find('.');
   size_t nch = name.length();
   Bool_t dotlast = kFALSE;
   if (nch && name[nch-1] == '.') dotlast = kTRUE;
   TBranchElement *branch = new TBranchElement(name.c_str(),sinfo,id,objadd,bufsize,splitlevel);
   fBranches.Add(branch);
   if (splitlevel > 0) {
      // Loop on all public data members of the class and its base classes
      TObjArray *blist = branch->GetListOfBranches();
      TIter next(sinfo->GetElements());
      TStreamerElement *element;
      id = 0;
      while ((element = (TStreamerElement*)next())) {
         std::string bname(name);
         char *pointer = (char*)objadd + element->GetOffset();
         Bool_t isBase = element->IsA() == TStreamerBase::Class();
         if (isBase) {
            TClass *clbase = element->GetClassPointer();
            if (clbase == TObject::Class() && cl->CanIgnoreTObjectStreamer()) continue;
         }
         if ( dot != std::string::npos ) {
            if (dotlast) {
               bname += element->GetFullName();
            } else if ( !isBase) {
               bname += ".";
               bname += element->GetFullName();
            }
         } else {
            bname = element->GetFullName();
         }
         TBranchElement *bre = new TBranchElement(bname.c_str(),sinfo,id,pointer,bufsize,splitlevel-1);
         blist->Add(bre);
         id++;
      }
   }
   branch->SetAddress(add);

   if (delobj) {delete objadd; *ppointer=0;}
   return branch;
}


//______________________________________________________________________________
void TBranchElement::Browse(TBrowser *b)
{
   // Browse the branch content.

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches > 0) {
      TList persistentBranches;
      TBranch* branch=0;
      TIter iB(&fBranches);
      while((branch=(TBranch*)iB())) {
         if (branch->IsFolder()) persistentBranches.Add(branch);
         else {
            // only show branches corresponding to persistent members
            TClass* cl=0;
            if (strlen(GetClonesName()))
               // this works both for top level branches and for sub-branches,
               // as GetClonesName() is properly updated for sub-branches
               cl=gROOT->GetClass(GetClonesName());
            else {
               cl=gROOT->GetClass(GetClassName());

               // check if we're in a sub-branch of this class
               // we can only find out asking the streamer given our ID
               ULong_t *elems=0;
               TStreamerElement *element=0;
               TClass* clsub=0;
               if (fID>=0 && GetInfo()
                   && ((elems=GetInfo()->GetElems()))
                   && ((element=(TStreamerElement *)elems[fID]))
                   && ((clsub=element->GetClassPointer())))
                  cl=clsub;
            }
            if (cl) {
               TString strMember=branch->GetName();
               Size_t mempos=strMember.Last('.');
               if (mempos!=kNPOS)
                  strMember.Remove(0, (Int_t)mempos+1);
               mempos=strMember.First('[');
               if (mempos!=kNPOS)
                  strMember.Remove((Int_t)mempos);
               TDataMember* m=cl->GetDataMember(strMember);
               if (!m || m->IsPersistent()) persistentBranches.Add(branch);
            } else persistentBranches.Add(branch);
         } // branch if not a folder
      }
      persistentBranches.Browse(b);
      // add all public const methods without params
      if (GetBrowsables() && GetBrowsables()->GetSize())
         GetBrowsables()->Browse(b);
   } else {
      if (GetBrowsables() && GetBrowsables()->GetSize()) {
         GetBrowsables()->Browse(b);
         return;
      }
      // Get the name and strip any extra brackets
      // in order to get the full arrays.
      TString slash("/"), escapedSlash("\\/");
      TString name = GetName();
      Int_t pos = name.First('[');
      if (pos!=kNPOS) name.Remove(pos);

      TString mothername;
      if (GetMother()) {
         mothername = GetMother()->GetName();
         pos = mothername.First('[');
         if (pos!=kNPOS) mothername.Remove(pos);

         Int_t len = mothername.Length();
         if (len) {
            if (mothername(len-1)!='.') {
               // We do not know for sure whether the mother's name is
               // already preprended.  So we need to check:
               //    a) it is prepended
               //    b) it is NOT the name of a daugher (i.e. mothername.mothername exist)
               TString doublename = mothername;
               doublename.Append(".");
               Int_t isthere = (name.Index(doublename)==0);
               if (!isthere) {
                  name.Prepend(doublename);
               } else {
                  if (GetMother()->FindBranch(mothername)) {
                     doublename.Append(mothername);
                     isthere = (name.Index(doublename)==0);
                     if (!isthere) {
                        mothername.Append(".");
                        name.Prepend(mothername);
                     }
                  } else {
                     // Nothing to do because the mother's name is
                     // already in the name.
                  }
               }
            } else {
               // If the mother's name end with a dot then
               // the daughter probably already contains the mother's name
               if (name.Index(mothername)==kNPOS) {
                  name.Prepend(mothername);
               }
            }
         }
      }
      name.ReplaceAll(slash, escapedSlash);
      GetTree()->Draw(name, "", b ? b->GetDrawOption() : "");
      if (gPad) gPad->Update();
   }
}


//______________________________________________________________________________
void TBranchElement::BuildTitle(const char *name)
{
   //set branch/leaf name/title in case of a TClonesArray sub-branch

   char branchname[kMaxLen];
   Int_t nbranches = fBranches.GetEntries();
   for (Int_t i=0;i<nbranches;i++) {
      TBranchElement *bre = (TBranchElement*)fBranches.At(i);
      R__ASSERT(fType==3 || fType==4 );
      bre->SetType(fType*10+1);
      bre->fCollProxy = GetCollectionProxy();
      bre->BuildTitle(name);
      const char *fin = strrchr(bre->GetTitle(),'.');
      if (fin == 0) continue;
      bre->SetBranchCount(this); //primary branchcount
      TLeafElement *lf = (TLeafElement*)bre->GetListOfLeaves()->At(0);
      //if branch name is of the form fTracks.fCovar[3][4]
      //set the title to fCovar[fTracks_]

      strcpy(branchname,fin+1);
      char *dim = (char*)strstr(branchname,"[");
      Int_t nch = strlen(branchname);
      if (dim) {
         *dim = 0;
         nch = dim-branchname;
      }
      sprintf(branchname+nch,"[%s_]",name);

      bre->SetTitle(branchname);
      if (lf) lf->SetTitle(branchname);

      // is there a secondary branchcount ?
      //fBranchCount2 points to the secondary branchcount
      //in case a TClonesArray element has itself a branchcount.
      //Example in Event class with TClonesArray fTracks of Track objects.
      //if the Track object has two members
      //  Int_t    fNpoint;
      //  Float_t *fPoints;  //[fNpoint]
      //In this case the TBranchElement fTracks.fPoints has
      // -its primary branchcount pointing to the branch fTracks
      // -its secondary branchcount pointing to fTracks.fNpoint
      Int_t stype = bre->GetStreamerType();
      if (stype > 40 && stype < 61) {
         char name2[kMaxLen];
         strcpy(name2,bre->GetName());
         char *bn = strrchr(name2,'.');
         if (!bn) continue;
         TStreamerBasicPointer *el = (TStreamerBasicPointer*)bre->GetInfo()->GetElements()->FindObject(bn+1);
         strcpy(bn+1,el->GetCountName());
         TBranchElement *bc2 = (TBranchElement*)fBranches.FindObject(name2);
         bre->SetBranchCount2(bc2);
         //printf("Branch:%s has a secondary branchcount, bc2=%s\n",bre->GetName(),bc2->GetName());
      }
   }
}


//______________________________________________________________________________
Int_t TBranchElement::Fill()
{
//*-*-*-*-*-*-*-*Loop on all leaves of this branch to fill Basket buffer*-*-*
//*-*            =======================================================
//
// The function returns the number of bytes committed to the
// individual branch(es).
// If a write error occurs, the number of bytes returned is -1.
// If no data are written, because e.g. the branch is disabled,
// the number of bytes returned is 0.
//

   Int_t nbytes = 0, nwrite = 0, nerror = 0;
   Int_t nbranches = fBranches.GetEntriesFast();
   // update addresses if top level branch
   if (fID < 0) {
      if (!fAddress) {
         Error("Fill","attempt to fill branch %s while addresss is not set",GetName());
         return 0;
      }
      void *add1  = fObject;
      void **add2 = (void**)fAddress;
      if (add1 != *add2) {
         SetAddress(fAddress);
      }

   }

   //if the tree has a TRefTable, set the current branch if branch is not a basic type
   if (fType >= 0 && fType < 10) {
      TBranchRef *bref = fTree->GetBranchRef();
      if (bref) bref->SetParent(this);
   }

   if (nbranches) {
      if (fType == 3 || fType == 4)  {
         nbytes += (nwrite = TBranch::Fill());  //TClonesArray counter
         if ( nwrite < 0 )  {
            Error("Fill","Failed filling branch:%s, nbytes=%d",GetName(),nwrite);
            nerror++;
         }
      }
      else  {
         fEntries++;
      }
      for (Int_t i=0;i<nbranches;i++)  {
         TBranchElement *branch = (TBranchElement*)fBranches[i];
         if (!branch->TestBit(kDoNotProcess))  {
            nbytes += (nwrite = branch->Fill());
            if ( nwrite < 0 )  {
               Error("Fill","Failed filling branch:%s.%s, nbytes=%d",GetName(),branch->GetName(),nwrite);
               nerror++;
            }
         }
      }
   } else {
      if (!TestBit(kDoNotProcess)) {
         nbytes += (nwrite = TBranch::Fill());
         if ( nwrite < 0 )  {
            Error("Fill","Failed filling branch:%s, nbytes=%d",GetName(),nwrite);
            nerror++;
         }
      }
   }
   if (fTree->Debug() > 0) {
      Long64_t entry = fEntries;
      if (entry >= fTree->GetDebugMin() && entry <= fTree->GetDebugMax()) {
         printf("Fill: %lld, branch=%s, nbytes=%d\n",entry,GetName(),nbytes);
      }
   }
   return nerror==0 ? nbytes : -1;
}

//______________________________________________________________________________
void TBranchElement::FillLeaves(TBuffer &b)
{
//  Fill buffers of this branch

   if (!fObject) return;
   if (fType <= 2 && TestBit(kBranchObject)) b.MapObject((TObject*)fObject);

   if (fType == 4) {           // STL vector/list of objects
      if (!fObject) {
         b << 0;
      } else {
         TVirtualCollectionProxy::TPushPop helper(GetCollectionProxy(),fObject);
         Int_t n = fCollProxy->Size();
         if (n > fMaximum) fMaximum = n;
         b << n;
      }
   } else if (fType == 41) {   // sub branch of an STL class

      //char **ppointer = (char**)fAddress;
      if (!fObject) return;
      TVirtualCollectionProxy::TPushPop helper(GetCollectionProxy(),fObject);// NOTE: this wont work if a pointer to vector is split!
      Int_t n = fCollProxy->Size();
      fInfo->WriteBufferSTL(b,fCollProxy,n,fID,fOffset);

   } else if (fType == 3) {   //top level branch of a TClonesArray
      if (fTree->GetMakeClass()) {
         TClass *cl = gROOT->GetClass(GetClonesName());
         cl->GetStreamerInfo()->ForceWriteInfo((TFile *)b.GetParent());
         Int_t *n = (Int_t*)fAddress;
         b << *n;
         return;
      }
      TClonesArray *clones = (TClonesArray*)fObject;
      if (!clones) {
         b << 0;
      } else {
         Int_t n = clones->GetEntriesFast();
         if (n > fMaximum) fMaximum = n;
         b << n;
      }
   } else if (fType == 31) {   // sub branch of a TClonesArray
      if (fTree->GetMakeClass()) {
         Int_t atype = fStreamerType;
         if (atype > 54) return;
         if (!fAddress) return;
         Int_t *nn = (Int_t*)fBranchCount->GetAddress();
         Int_t n = *nn;
         if (atype>40) {
            printf("Clonesa: %s, n=%d, sorry not supported yet\n",GetName(),n);
         }
         if (atype > 20) {
            atype -= 20;
            TLeafElement *leaf = (TLeafElement*)fLeaves.UncheckedAt(0);
            n *= leaf->GetLenStatic();
         }
         switch (atype) {
            case  1: {b.WriteFastArray((Char_t*)   fAddress, n); break;}
            case  2: {b.WriteFastArray((Short_t*)  fAddress, n); break;}
            case  3: {b.WriteFastArray((Int_t*)    fAddress, n); break;}
            case  4: {b.WriteFastArray((Long_t*)   fAddress, n); break;}
            case  5: {b.WriteFastArray((Float_t*)  fAddress, n); break;}
            case  6: {b.WriteFastArray((Int_t*)    fAddress, n); break;}
            case  8: {b.WriteFastArray((Double_t*) fAddress, n); break;}
            case 11: {b.WriteFastArray((UChar_t*)  fAddress, n); break;}
            case 12: {b.WriteFastArray((UShort_t*) fAddress, n); break;}
            case 13: {b.WriteFastArray((UInt_t*)   fAddress, n); break;}
            case 14: {b.WriteFastArray((ULong_t*)  fAddress, n); break;}
            case 15: {b.WriteFastArray((UInt_t*)   fAddress, n); break;}
            case 16: {b.WriteFastArray((Long64_t*) fAddress, n); break;}
            case 17: {b.WriteFastArray((ULong64_t*)fAddress, n); break;}
            case 18: {b.WriteFastArray((Bool_t*)   fAddress, n); break;}
            case  9: {
                        Double_t *xx = (Double_t*)fAddress;
                        for (Int_t ii=0;ii<n;ii++) b << (Float_t)xx[ii];
                        break;
                     }
         }
         return;
      }
      TClonesArray *clones = (TClonesArray*)fObject;
      if (!clones) return;
      Int_t n = clones->GetEntriesFast();
      fInfo->WriteBufferClones(b,clones,n,fID,fOffset);
   } else if (fType <= 2) {
      Int_t n = fInfo->WriteBufferAux(b,&fObject,fID,1,0,0); // NOTE: expanded
      if (fStreamerType == 6) {
         if (n > fMaximum) fMaximum = n;
      }
   }
}

//______________________________________________________________________________
Int_t TBranchElement::GetDataMemberOffset(const TClass *cl, const char *name)
{
// This function is for internal use only!
// Return the offset if 'name' is found in the list of real data for cl
// Output an error message if the 'name' is NOT found (to prevent reliance on the previous
// implementation.

// The previous implementation of the class had the following comments:

// Return the offset if 'name' is found in the list of real data for cl
// return offset od member name in class cl
// check for the following cases Otto and Axel
//
// Return the opposite of the offset if name is of the form 'XXX.YYY' where
//    XXX is a name found in the list of real data for cl
//
// case Otto
//    class TUsrSevtData2:public TMrbSubevent_Caen {
//    class TMrbSubevent_Caen:public TObject {
//       TUsrHitBuffer fHitBuffer;
//    class TUsrHitBuffer:public TObject {
//       Int_t fHighWater;
//       TClonesArray *fHits;
//    code below to get the correct address for fHitBuffer.fHits
//
//    i.e. this is the case where we have a TClonesArray inside an object
//    which is embedded into another (which is stored in a branch)
//
// case Axel
//    class jet: public TLorentzVector {
//    TClonesArray* caJet=new TClonesArray("jet");
//    TTree* tree=new TTree("test","test",99);
//    tree->Branch("jet", "TClonesArray",&caJet, 32000);
//
//    i.e this is the case where we have an embedded object inside an object
//    stored in a TClonesArray

   Int_t offset = 0;
   TRealData *rd = cl->GetRealData(name);

   if (!rd) {
      Error("GetDataMemberOffset","obsolete call with (%s,%s)\n",
            cl->GetName(),name);
   } else {
      offset = rd->GetThisOffset();
   }

   if (gDebug > 3) {
      printf("GetDataMemberOffset(%s,%s) => %d\n",
             cl->GetName(),name,offset);
   }
   return offset;
}

//______________________________________________________________________________
void TBranchElement::SetupAddresses()
{
   // If the branch address had not yet been set,
   // we set all addresses starting with the top level parent branch

   // This is requires to be done in order for GetOffset
   // to be guarantee correct and for GetEntry to be run.
   if (fAddress == 0 && fTree->GetMakeClass() == 0) {
      if (TestBit(kDoNotProcess)) return;
      TBranchElement *mother = (TBranchElement*)GetMother();
      TClass *cl = gROOT->GetClass(mother->GetClassName());
      if (fInfo && fInfo->GetOffsets()) fInfo->BuildOld();
      if (!mother || !cl) return;
      if (!mother->GetAddress()) {
         Bool_t motherStatus = mother->TestBit(kDoNotProcess);
         mother->ResetBit(kDoNotProcess);
         mother->SetAddress(0);
         mother->SetBit(kDoNotProcess,motherStatus);
      }
   }
}

//______________________________________________________________________________
Int_t TBranchElement::GetEntry(Long64_t entry, Int_t getall)
{
//*-*-*-*-*Read all branches of a BranchElement and return total number of bytes
//*-*      ====================================================================
//   If entry = 0 take current entry number + 1
//   If entry < 0 reset entry number to 0
//
//  The function returns the number of bytes read from the input buffer.
//  If entry does not exist  the function returns 0.
//  If an I/O error occurs,  the function returns -1.
//
//  See IMPORTANT REMARKS in TTree::GetEntry

   Int_t nbranches = fBranches.GetEntriesFast();

   Int_t nbytes = 0;

   if (fAddress == 0 && fTree->GetMakeClass() == 0) {
      SetupAddresses();
   }

   if (nbranches) {
      //branch has daughters
      //one must always read the branch counter.
      //In the case when one reads consecutively twice the same entry,
      //the user may have cleared the TClonesArray between the 2 GetEntry
      if ( fType == 3 || fType == 4 )  {
         fReadEntry = entry;
         nbytes += TBranch::GetEntry(entry, getall);
      }
      Int_t i;
      switch(fSTLtype)  {
         case TClassEdit::kSet:
         case TClassEdit::kMultiSet:
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
            break;
         default:
            for (i=0;i<nbranches;i++)  {
               TBranch *branch = (TBranch*)fBranches[i];
               Int_t    nb = branch->GetEntry(entry, getall);
               if (nb < 0) return nb;
               nbytes += nb;
            }
            break;
      }
   } else {
      //terminal branch
      if (fBranchCount && fBranchCount->GetReadEntry() != entry) nbytes += fBranchCount->TBranch::GetEntry(entry,getall);
      nbytes += TBranch::GetEntry(entry, getall);
   }

   // if Tree has a TBranchRef, set the ReadEntry in the TBranchRef
   TBranchRef *bref = fTree->GetBranchRef();
   if (bref) {
      bref->SetParent(this);
      bref->SetReadEntry(entry);
   }

   if (fTree->Debug() > 0) {
      if (entry >= fTree->GetDebugMin() && entry <= fTree->GetDebugMax()) {
         printf("GetEntry: %lld, branch=%s, nbytes=%d\n",entry,GetName(),nbytes);
      }
   }
   return nbytes;
}

//______________________________________________________________________________
const char *TBranchElement::GetIconName() const
{
   // Return icon name depending on type of branch element.

   if (IsFolder())
      return "TBranchElement-folder";
   else
      return "TBranchElement-leaf";
}

//______________________________________________________________________________
Bool_t TBranchElement::CheckBranchID()
{
   // Need to reassign branches in case schema evolution has scrambled leaf list.

   if ( GetID() >= 0 ) {
      size_t pos;
      std::string s( GetName() );
      pos = s.rfind('.');
      if ( pos != std::string::npos )  {
         s = s.substr(pos+1);
      }
      while ( (pos=s.rfind('[')) != std::string::npos ) {
         s = s.substr(0,pos);
      }
      int offset = 0;
      TStreamerElement* elt = fInfo->GetStreamerElement(s.c_str(),offset);
      if ( elt )   {
         size_t ndata = fInfo->GetNdata();
         ULong_t *elems = fInfo->GetElems();

         for(size_t i=0; i < ndata; ++i )  {
            if ( (TStreamerElement*)elems[i] == elt )  {
               fID = i;
               break;
            }
         }
      }
      else  {
         // Element may be missing, if data member got removed.
         // Warning("CheckBranchID","Cannot find streamer element:%s",s.c_str());
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
TStreamerInfo *TBranchElement::GetInfo()
{
  //return pointer to TStreamerinfo object for the class of this branch
  //rebuild the info if not yet done

   Bool_t optim = TStreamerInfo::CanOptimize();
   if (fInfo) {
      if (!fInfo->GetOffsets()) {
         TStreamerInfo::Optimize(kFALSE);
         fInfo->Compile();
         TStreamerInfo::Optimize(optim);
      }
      if (!fInit) fInit = CheckBranchID();
      return fInfo;
   }
   TClass *cl = fBranchClass;
   if (cl) {
      TStreamerInfo::Optimize(kFALSE);
      if (cl == TClonesArray::Class()) fClassVersion = TClonesArray::Class()->GetClassVersion();
      fInfo = cl->GetStreamerInfo(fClassVersion);
      if (fCheckSum != 0 && (cl->IsForeign() ||
                             (!cl->IsLoaded() &&
                              fClassVersion==1 &&
                              cl->GetStreamerInfos()->At(1)!=0 &&
                              fCheckSum!= ((TStreamerInfo*)cl->GetStreamerInfos()->At(1))->GetCheckSum()))) {
         Int_t ninfos = cl->GetStreamerInfos()->GetEntriesFast();
         for (Int_t i=1;i<ninfos;i++) {
            TStreamerInfo *info = (TStreamerInfo*)cl->GetStreamerInfos()->At(i);
            if (!info) continue;
            if (info->GetCheckSum() == fCheckSum) {
               fClassVersion = i;
               fInfo = cl->GetStreamerInfo(fClassVersion);
               break;
            }
         }
      }
      if (fInfo && !fInfo->GetOffsets()) {
         fInfo->Compile();
      }
      TStreamerInfo::Optimize(optim);
      if ( !fInit ) fInit = CheckBranchID();
   }
   return fInfo;
}

//______________________________________________________________________________
Int_t TBranchElement::GetMaximum() const
{
// Return maximum count value of the branchcount if any

   if (fBranchCount) return fBranchCount->GetMaximum();
   return fMaximum;
}

//______________________________________________________________________________
const char *TBranchElement::GetTypeName() const
{
   // return type name of element in the branch

   if (fType == 3  || fType == 4) {
      return "Int_t";
   }
   if (fStreamerType <=0 || fStreamerType >= 60) return fClassName.Data();
   const char *types[19] = {"",
                            "Char_t","Short_t","Int_t","Long_t","Float_t",
                            "Int_t","",
                            "Double_t","Double32_t",
                            "",
                            "UChar_t","UShort_t","UInt_t","ULong_t","UInt_t",
                            "Long64_t","ULong64_t","Bool_t"};
   Int_t itype = fStreamerType%20;
   return types[itype];
}

//______________________________________________________________________________
Double_t TBranchElement::GetValue(Int_t j, Int_t len, Bool_t subarr) const
{
// Returns branch value. If the leaf is an array, j is the index in the array
// If leaf is an array inside a TClonesArray, len should be the length of the
// array.  If subarr is true, then len is actually the index within the sub-array

   if (j == 0 && fBranchCount) {
      Int_t entry = fTree->GetReadEntry();
      fBranchCount->TBranch::GetEntry(entry);
      if (fBranchCount2) fBranchCount2->TBranch::GetEntry(entry);
   }

   if (fTree->GetMakeClass()) {
      if (!fAddress) return 0;
      if (fType == 3 || fType == 4 ) {    //top level branch of a TClonesArray
         return (Double_t)fNdata;
      } else if (fType == 31 || fType == 41) {    // sub branch of a TClonesArray
         Int_t atype = fStreamerType;
         if (atype < 20) atype += 20;
         return fInfo->GetValue(fAddress,atype,j,1);
      } else if (fType <= 2) {     // branch in split mode
         if (fStreamerType > 40 && fStreamerType < 55) {
            Int_t atype = fStreamerType - 20;
            return fInfo->GetValue(fAddress,atype,j,1);
         } else {
            return fInfo->GetValue(fObject,fID,j,-1);
         }
      }
   }

   if (fType == 31) {
      TClonesArray *clones = (TClonesArray*)fObject;
      if (subarr) return fInfo->GetValueClones(clones,fID, j, len,fOffset);
      else return fInfo->GetValueClones(clones,fID, j/len, j%len,fOffset);
   }  else if (fType == 41) {
      TVirtualCollectionProxy::TPushPop helper(fCollProxy,fObject);
      if (subarr) return fInfo->GetValueSTL(fCollProxy,fID, j, len,fOffset);
      else return fInfo->GetValueSTL(fCollProxy,fID, j/len, j%len,fOffset);
   } else {
      if (fInfo) return fInfo->GetValue(fObject,fID,j,-1);
      return 0;
   }
}

//______________________________________________________________________________
void *TBranchElement::GetValuePointer() const
{
// Returns pointer to first data element of this branch
// Currently used only for members of type character

   if (fBranchCount) {
      Int_t entry = fTree->GetReadEntry();
      fBranchCount->TBranch::GetEntry(entry);
      if (fBranchCount2) fBranchCount2->TBranch::GetEntry(entry);
   }
   if (fTree->GetMakeClass()) {
      if (!fAddress) return 0;
      if (fType == 3) {    //top level branch of a TClonesArray
         //return &fNdata;
         return 0;
      } else if (fType == 4) {    //top level branch of a TClonesArray
         //return &fNdata;
         return 0;
      } else if (fType == 31) {    // sub branch of a TClonesArray
         //Int_t atype = fStreamerType;
         //if (atype < 20) atype += 20;
         //return fInfo->GetValue(fAddress,atype,j,1);
         return 0;
      } else if (fType == 41) {    // sub branch of a TClonesArray
         //Int_t atype = fStreamerType;
         //if (atype < 20) atype += 20;
         //return fInfo->GetValue(fAddress,atype,j,1);
         return 0;
      } else if (fType <= 2) {     // branch in split mode
         if (fStreamerType > 40 && fStreamerType < 55) {
            //Int_t atype = fStreamerType - 20;
            //return fInfo->GetValue(fAddress,atype,j,1);
            return 0;
         } else {
            //return fInfo->GetValue(fObject,fID,j,-1);
            return 0;
         }
      }
   }

   if (fType == 31) {
      //TClonesArray *clones = (TClonesArray*)fObject;
      //if (subarr) return fInfo->GetValueClones(clones,fID, j, len,fOffset);
      //else return fInfo->GetValueClones(clones,fID, j/len, j%len,fOffset);
      return 0;
   } else if (fType == 41) {
      //TClonesArray *clones = (TClonesArray*)fObject;
      //if (subarr) return fInfo->GetValueClones(clones,fID, j, len,fOffset);
      //else return fInfo->GetValueClones(clones,fID, j/len, j%len,fOffset);
      return 0;
   } else {
      //return fInfo->GetValue(fObject,fID,j,-1);
      if (!fInfo || !fObject) return 0;
      char **val = (char**)(fObject+fInfo->GetOffsets()[fID]);
      return *val;
   }
}

//______________________________________________________________________________
Bool_t TBranchElement::IsFolder() const
{
   // Return TRUE if more than one leaf, FALSE otherwise.

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches >= 1) return kTRUE;
   TList* browsables= const_cast<TBranchElement*>(this)->GetBrowsables();
   return (browsables && browsables->GetSize());
}

//______________________________________________________________________________
Bool_t TBranchElement::IsMissingCollection() const
{
   // In version of ROOT older than 4.00/03, if a collection (TClonesArray
   // or stl container) was split but the pointer to the collection was zeroed
   // out, nothing was saved.  Hence there is no __easy__ way to detect the
   // case.  In newer version, a zero is inserted so that a 'missing' collection
   // appears as an empty collection.
   // This function helps in detecting this case so that we can recover nicely.

   Bool_t ismissing = kFALSE;

   TBasket *basket = (TBasket*)fBaskets.UncheckedAt(fReadBasket);
   if (basket && fTree) {
      Int_t entry = fTree->GetReadEntry();
      Int_t first  = fBasketEntry[fReadBasket];
      Int_t last;
      if (fReadBasket == fWriteBasket) last = fEntryNumber - 1;
      else                             last = fBasketEntry[fReadBasket+1] - 1;
      Int_t *entryOffset = basket->GetEntryOffset();
      Int_t bufbegin ;
      Int_t bufnext;
      if (entryOffset) {
         bufbegin = entryOffset[entry-first];

         if (entry<last) {
            bufnext = entryOffset[entry+1-first];
         } else {
            bufnext = basket->GetLast();
         }
         if (bufnext==bufbegin) {
            ismissing = kTRUE;
         } else {
            // fixed length buffer so this is not the case here.
            if (basket->GetNevBufSize()==0) {
               ismissing = kTRUE;
            }
         }
      }
   }
   return ismissing;
}

//______________________________________________________________________________
void TBranchElement::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print TBranch parameters*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   Int_t nbranches = fBranches.GetEntriesFast();
   if (strncmp(option,"debugAddress",strlen("debugAddress"))==0) {
      if (strlen(option)==strlen("debugAddress")) {
         Printf("%-24s %-16s %2s %4s %-16s %-16s %8s %8s %s\n",
                "Branch Name", "Streamer Class", "ID", "Type", "Class", "Parent", "pOffset", "fOffset", "fObject");
      }
      TBranchElement *parent = (TBranchElement*) GetMother()->GetSubBranch(this);
      Int_t ind = parent->GetListOfBranches()->IndexOf(this);
      if (strlen(GetName())>24) Printf("%-24s\n%-24s ", GetName(),"");
      else Printf("%-24s ", GetName());
      Printf("%-16s %2d %4d %-16s %-16s %8x %8x %8x\n",
             ((TBranchElement*)this)->GetInfo()->GetName(), GetID(), GetType(),
             GetClassName(), GetParentName(),
             (fBranchOffset&&parent) ? parent->fBranchOffset[ind] : 0,
             GetOffset(), GetObject());
      TObjArray* brl = ((TBranchElement*)this)->GetListOfBranches();
      Int_t nbranches = brl->GetEntriesFast();
      for (Int_t i = 0; i < nbranches; ++i) {
         TBranchElement* subbranch = (TBranchElement*)brl->At(i);
         subbranch->Print("debugAddressSub");
      }
      return;
   }
   if (nbranches) {
      if (fID == -2) {
         if (strcmp(GetName(),GetTitle()) == 0) {
            Printf("*Branch  :%-66s *",GetName());
         } else {
            Printf("*Branch  :%-9s : %-54s *",GetName(),GetTitle());
         }
         Printf("*Entries : %8d : BranchElement (see below)                              *",Int_t(fEntries));
         Printf("*............................................................................*");
      }
      if (fType >= 2) {
         TBranch::Print(option);
      }
      for (Int_t i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches.At(i);
         branch->Print(option);
      }
   } else {
      TBranch::Print(option);
   }
}

//______________________________________________________________________________
void TBranchElement::PrintValue(Int_t lenmax) const
{
   // Prints leaf value

   if (fTree->GetMakeClass()) {
      if (!fAddress) return;
      if (fType == 3 || fType == 4) {    //top level branch of a TClonesArray
         printf(" %-15s = %d\n",GetName(),fNdata);
         return;
      } else if (fType == 31 || fType == 41) {    // sub branch of a TClonesArray
         Int_t n = TMath::Min(10,fNdata);
         Int_t atype = fStreamerType+TStreamerInfo::kOffsetL;
         // TStreamerInfo::kOffsetL + TStreamerInfo::kChar is printed as a character strings
         // and hence could print weird character.  So let's do let damage and print an unsigned
         // char instead (not perfect but better).
         if (fStreamerType==TStreamerInfo::kChar) atype=TStreamerInfo::kOffsetL + TStreamerInfo::kUChar;
         if (atype > 54) {
            //more logic required here (like in ReadLeaves)
            printf(" %-15s = %d\n",GetName(),fNdata);
            return;
         }
         if (fStreamerType > 20) {
            atype -= 20;
            TLeafElement *leaf = (TLeafElement*)fLeaves.UncheckedAt(0);
            n *= leaf->GetLenStatic();
         }
         if (fInfo) fInfo->PrintValue(GetName(),fAddress,atype,n,lenmax);
         return;
      } else if (fType <= 2) {     // branch in split mode
         if (fStreamerType > 40 && fStreamerType < 55) {
            Int_t atype = fStreamerType - 20;
            Int_t n = (Int_t)((TBranchElement*)fBranchCount)->GetValue(0,0);
            if (fInfo) fInfo->PrintValue(GetName(),fAddress,atype,n,lenmax);
         } else {
            if (fInfo) fInfo->PrintValue(GetName(),fObject,fID,-1,lenmax);
         }
         return;
      }
      return;
   }
   if (fType == 3) {
      printf(" %-15s = %d\n",GetName(),fNdata);
   } else if (fType == 31) {
      TClonesArray *clones = (TClonesArray*)fObject;
      if (fInfo) fInfo->PrintValueClones(GetName(),clones,fID,fOffset,lenmax);
   } else if (fType == 41) {
      TVirtualCollectionProxy::TPushPop helper(fCollProxy,fObject); // Object);
      if (fInfo) fInfo->PrintValueSTL(GetName(),fCollProxy,fID,fOffset,lenmax);
   } else {
      if (fInfo) fInfo->PrintValue(GetName(),fObject,fID,-1,lenmax);
   }
}

//______________________________________________________________________________
void TBranchElement::ReadLeaves(TBuffer &b)
{
   // Read buffers for this branch

   if (fTree->GetMakeClass()) {
      if (fType == 3 || fType == 4) {    //top level branch of a TClonesArray
         Int_t *n = (Int_t*)fAddress;
         b >> n[0];
         if (n[0]<0 || n[0]>fMaximum) {
            if (IsMissingCollection()) {
               n[0] = 0;
               b.SetBufferOffset( b.Length() - sizeof(n) );
            } else {
               Error("ReadLeaves",
                     "Incorrect size read for the container in %s\nThe size read is %d when the maximum is %d\nThe size is reset to 0 for this entry (%d)",
                     GetName(),n,fMaximum,GetReadEntry());
               n[0] = 0;
            }
         }
         fNdata = n[0];
         if ( fType == 4)   {
            Int_t i, nbranches = fBranches.GetEntriesFast();
            switch(fSTLtype)  {
               case TClassEdit::kSet:
               case TClassEdit::kMultiSet:
               case TClassEdit::kMap:
               case TClassEdit::kMultiMap:
                  for (i=0;i<nbranches;i++)  {
                     TBranch *branch = (TBranch*)fBranches[i];
                     Int_t    nb = branch->GetEntry(GetReadEntry(), 1);
                     if (nb < 0) break;
                  }
                  break;
               default:
                  break;
            }
         }
         return;
      } else if (fType == 31 || fType == 41) {    // sub branch of a TClonesArray
         fNdata = fBranchCount->GetNdata();
         Int_t atype = fStreamerType;
         if (atype > 54) return;
         if (!fAddress) return;
         Int_t n = fNdata;
         if (atype>40) {
            atype -= 40;
            if (!fBranchCount2) return;
            const char *len_where = (char*)fBranchCount2->fAddress;
            if (!len_where) return;
            Int_t len_atype = fBranchCount2->fStreamerType;
            Int_t length;
            Int_t k;
            Char_t isArray;
            for( k=0; k<n; k++) {
               char **where = &(((char**)fAddress)[k]);
               delete [] *where;
               *where = 0;
               switch(len_atype) {
                  case  1:  {length = ((Char_t*)   len_where)[k]; break;}
                  case  2:  {length = ((Short_t*)  len_where)[k]; break;}
                  case  3:  {length = ((Int_t*)    len_where)[k]; break;}
                  case  4:  {length = ((Long_t*)   len_where)[k]; break;}
                     //case  5:  {length = ((Float_t*) len_where)[k]; break;}
                  case  6:  {length = ((Int_t*)    len_where)[k]; break;}
                   //case  8:  {length = ((Double_t*)len_where)[k]; break;}
                  case 11:  {length = ((UChar_t*)  len_where)[k]; break;}
                  case 12:  {length = ((UShort_t*) len_where)[k]; break;}
                  case 13:  {length = ((UInt_t*)   len_where)[k]; break;}
                  case 14:  {length = ((ULong_t*)  len_where)[k]; break;}
                  case 15:  {length = ((UInt_t*)   len_where)[k]; break;}
                  case 16:  {length = ((Long64_t*) len_where)[k]; break;}
                  case 17:  {length = ((ULong64_t*)len_where)[k]; break;}
                  case 18:  {length = ((Bool_t*)   len_where)[k]; break;}
                  default: continue;
               }
               b >> isArray;
               if (length <= 0)  continue;
               if (isArray == 0) continue;
               switch (atype) {
                  case  1:  {*where=new char[sizeof(Char_t)*length]; b.ReadFastArray((Char_t*) *where, length); break;}
                  case  2:  {*where=new char[sizeof(Short_t)*length]; b.ReadFastArray((Short_t*) *where, length); break;}
                  case  3:  {*where=new char[sizeof(Int_t)*length]; b.ReadFastArray((Int_t*)   *where, length); break;}
                  case  4:  {*where=new char[sizeof(Long_t)*length]; b.ReadFastArray((Long_t*)  *where, length); break;}
                  case  5:  {*where=new char[sizeof(Float_t)*length]; b.ReadFastArray((Float_t*) *where, length); break;}
                  case  6:  {*where=new char[sizeof(Int_t)*length]; b.ReadFastArray((Int_t*)   *where, length); break;}
                  case  8:  {*where=new char[sizeof(Double_t)*length]; b.ReadFastArray((Double_t*)*where, length); break;}
                  case 11:  {*where=new char[sizeof(UChar_t)*length]; b.ReadFastArray((UChar_t*) *where, length); break;}
                  case 12:  {*where=new char[sizeof(UShort_t)*length]; b.ReadFastArray((UShort_t*)*where, length); break;}
                  case 13:  {*where=new char[sizeof(UInt_t)*length]; b.ReadFastArray((UInt_t*)  *where, length); break;}
                  case 14:  {*where=new char[sizeof(ULong_t)*length]; b.ReadFastArray((ULong_t*) *where, length); break;}
                  case 15:  {*where=new char[sizeof(UInt_t)*length]; b.ReadFastArray((UInt_t*)  *where, length); break;}
                  case 16:  {*where=new char[sizeof(Long64_t)*length]; b.ReadFastArray((Long64_t*)  *where, length); break;}
                  case 17:  {*where=new char[sizeof(ULong64_t)*length]; b.ReadFastArray((ULong64_t*)*where, length); break;}
                  case 18:  {*where=new char[sizeof(Bool_t)*length]; b.ReadFastArray((Bool_t*) *where, length); break;}
               }
            }
            return;
         }
         if (atype > 20) {
            atype -= 20;
            TLeafElement *leaf = (TLeafElement*)fLeaves.UncheckedAt(0);
            n *= leaf->GetLenStatic();
         }
         switch (atype) {
            case  1: {b.ReadFastArray((Char_t*)  fAddress, n); break;}
            case  2: {b.ReadFastArray((Short_t*) fAddress, n); break;}
            case  3: {b.ReadFastArray((Int_t*)   fAddress, n); break;}
            case  4: {b.ReadFastArray((Long_t*)  fAddress, n); break;}
            case  5: {b.ReadFastArray((Float_t*) fAddress, n); break;}
            case  6: {b.ReadFastArray((Int_t*)   fAddress, n); break;}
            case  8: {b.ReadFastArray((Double_t*)fAddress, n); break;}
            case 11: {b.ReadFastArray((UChar_t*) fAddress, n); break;}
            case 12: {b.ReadFastArray((UShort_t*)fAddress, n); break;}
            case 13: {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
            case 14: {b.ReadFastArray((ULong_t*) fAddress, n); break;}
            case 15: {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
            case 16: {b.ReadFastArray((Long64_t*)fAddress, n); break;}
            case 17: {b.ReadFastArray((ULong64_t*)fAddress, n); break;}
            case 18: {b.ReadFastArray((Bool_t*)  fAddress, n); break;}
            case  9: {
                        Double_t *xx = (Double_t*)fAddress;
                        Float_t afloat;
                        for (Int_t ii=0;ii<n;ii++) {
                           b >> afloat; xx[ii] = Double_t(afloat);
                        }
                        break;
                     }
         }
         return;
      } else if (fType <= 2) {     // branch in split mode
         if (fStreamerType > 40 && fStreamerType < 55) {
            Int_t atype = fStreamerType - 40;
            Int_t n = (Int_t)fBranchCount->GetValue(0,0);
            fNdata = n;
            Char_t isArray;
            b >> isArray;
            switch (atype) {
               case  1: {b.ReadFastArray((Char_t*)  fAddress, n); break;}
               case  2: {b.ReadFastArray((Short_t*) fAddress, n); break;}
               case  3: {b.ReadFastArray((Int_t*)   fAddress, n); break;}
               case  4: {b.ReadFastArray((Long_t*)  fAddress, n); break;}
               case  5: {b.ReadFastArray((Float_t*) fAddress, n); break;}
               case  6: {b.ReadFastArray((Int_t*)   fAddress, n); break;}
               case  8: {b.ReadFastArray((Double_t*)fAddress, n); break;}
               case 11: {b.ReadFastArray((UChar_t*) fAddress, n); break;}
               case 12: {b.ReadFastArray((UShort_t*)fAddress, n); break;}
               case 13: {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
               case 14: {b.ReadFastArray((ULong_t*) fAddress, n); break;}
               case 15: {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
               case 16: {b.ReadFastArray((Long64_t*) fAddress, n); break;}
               case 17: {b.ReadFastArray((ULong64_t*)fAddress, n); break;}
               case 18: {b.ReadFastArray((Bool_t*)   fAddress, n); break;}
               case  9: {
                           Double_t *xx = (Double_t*)fAddress;
                           Float_t afloat;
                           for (Int_t ii=0;ii<n;ii++) {
                              b>> afloat; xx[ii] = Double_t(afloat);
                           }
                           break;
                        }
            }
         } else {
            fNdata = 1;
            if (fAddress) {
               //char **arr = &fObject;
               fInfo->ReadBuffer(b,(char**)&fObject,fID);
            } else {
               fNdata = 0;
            }
         }
         return;
      }
   }

   if (fType <=2 && TestBit(kBranchObject)) {
      b.MapObject((TObject*)fObject);
   }

   if (fType == 4) {           // STL vector/list of objects
      //Error("ReadLeaves","STL split mode not yet implemented (error 1)\n");
      Int_t n;
      b >> n;
      if (n<0 || n>fMaximum) {
         if (IsMissingCollection()) {
            n = 0;
            b.SetBufferOffset( b.Length() - sizeof(n) );
         } else {
            Error("ReadLeaves",
                  "Incorrect size read for the container in %s\n\tThe size read is %d while the maximum is %d\n\tThe size is reset to 0 for this entry (%d)",
                  GetName(),n,fMaximum,GetReadEntry());
            n = 0;
         }
      }
      fNdata = n;
      if (!fObject) return;
      if ( fType == 4)   {
         // Note: Proxy-helper needs to "embrace" the entire
         //       streaming of this STL container if the container
         //       is a set/multiset/map/multimap (what we do not
         //       know here).
         //       For vector/list/deque Allocate == Resize
         //                         and Commit   == noop.
         // TODO: Exception safety a la TPushPop
         TVirtualCollectionProxy* proxy = GetCollectionProxy();
         TVirtualCollectionProxy::TPushPop helper(proxy,fObject);
         void* env = proxy->Allocate(fNdata,true);
         Int_t i, nbranches = fBranches.GetEntriesFast();
         switch(fSTLtype)  {
            case TClassEdit::kSet:
            case TClassEdit::kMultiSet:
            case TClassEdit::kMap:
            case TClassEdit::kMultiMap:
               for (i=0;i<nbranches;i++)  {
                  TBranch *branch = (TBranch*)fBranches[i];
                  Int_t    nb = branch->GetEntry(GetReadEntry(), 1);
                  if (nb < 0) break;
               }
               break;
            default:
               break;
         }
         proxy->Commit(env);
      }
   } else if (fType == 41) {    // sub branch of an STL class
      //Error("ReadLeaves","STL split mode not yet implemented (error 2)\n");
      //char **ppointer = (char**)fAddress;
      fNdata = fBranchCount->GetNdata();
      if (!fObject) return;
      TVirtualCollectionProxy::TPushPop helper(GetCollectionProxy(),fObject);
      fInfo->ReadBufferSTL(b,fCollProxy,fNdata,fID,fOffset);
   } else if (fType == 3) {    //top level branch of a TClonesArray
      Int_t n;
      b >> n;
      if (n<0 || n>fMaximum) {
         if (IsMissingCollection()) {
            n = 0;
            b.SetBufferOffset( b.Length() - sizeof(n) );
         } else {
            Error("ReadLeaves",
                  "Incorrect size read for the container in %s\n\tThe size read is %d while the maximum is %d\n\tThe size is reset to 0 for this entry (%d)",
                  GetName(),n,fMaximum,GetReadEntry());
            n = 0;
         }
      }
      fNdata = n;
      TClonesArray *clones = (TClonesArray*)fObject;
      if (!clones) return;
      if (clones->IsZombie()) return;
      clones->Clear();
      clones->ExpandCreateFast(fNdata);
   } else if (fType == 31) {    // sub branch of a TClonesArray
      fNdata = fBranchCount->GetNdata();
      TClonesArray *clones = (TClonesArray*)fObject;
      if (!clones) return;
      if (clones->IsZombie()) return;
      fInfo->ReadBufferClones(b,clones,fNdata,fID,fOffset);
   } else if (fType <= 2) {     // branch in split mode
      if (fBranchCount) fNdata = (Int_t)fBranchCount->GetValue(0,0);
      else fNdata = 1;
      if (!fInfo) return;
      //char **arr = &fObject;
      fInfo->ReadBuffer(b,(char**)&fObject,fID);
      if (fStreamerType == 6) fNdata = (Int_t)GetValue(0,0);
   }
}

//______________________________________________________________________________
void TBranchElement::Reset(Option_t *option)
{
   // Reset a Branch.
   //
   //    Existing buffers are deleted
   //    Entries, max and min are reset
   //

   TBranch::Reset(option);
   fInfo = fBranchClass->GetStreamerInfo(fClassVersion);
   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->Reset(option);
   }
}

//______________________________________________________________________________
void TBranchElement::ResetAddress()
{
//*-*-*-*-*-*-*-*Reset the address of the branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ===============================
//

   fObject = 0;
   TBranch::ResetAddress();
}

//______________________________________________________________________________
namespace {
   void SwitchContainer(TObjArray *branches) {
      // Modify the container type of the branches

      const Int_t nbranches = branches->GetEntriesFast();
      for(Int_t i = 0; i < nbranches; ++i) {
         TBranchElement *br = (TBranchElement*)branches->At(i);
         switch ( br->GetType() ) {
            case 31: br->SetType(41); break;
            case 41: br->SetType(31); break;
         };
         SwitchContainer( br->GetListOfBranches());
      }
   }
}

//______________________________________________________________________________
TClass* TBranchElement::GetParentClass()
{
   // Return a pointer to the current type of the data member corresponding to branch element

   if ( fParentName.Data()[0] == 0 ) return 0;
   return fParentClass;
}

//______________________________________________________________________________
TClass* TBranchElement::GetCurrentClass()
{
   // Return a pointer to the current type of the data member corresponding to branch element

   TClass* cl = fCurrentClass;
   if ( cl ) return cl;

   // Get the current type of this data member!
   TStreamerInfo *brInfo = GetInfo();
   if (brInfo==0) {
      cl = gROOT->GetClass(GetClassName());
      R__ASSERT(cl && cl->GetCollectionProxy());
      fCurrentClass = cl;
      return cl;
   }
   TClass *motherCl = brInfo->GetClass();
   if (motherCl->GetCollectionProxy())   {
      cl = motherCl->GetCollectionProxy()->GetCollectionClass();
      if ( cl ) fCurrentClass = cl;
      return cl;
   }
   TStreamerElement *currentStreamerElement =
      ((TStreamerElement*)brInfo->GetElems()[GetID()]);
   TDataMember *dm =
      (TDataMember*)motherCl->GetListOfDataMembers()->FindObject(currentStreamerElement->GetName());

   TString newType;
   if ( dm==0 ) {
      // Either the class is not loaded or the data member is gone
      if (! motherCl->IsLoaded() ) {
         TStreamerInfo *newInfo = motherCl->GetStreamerInfo();
         if (newInfo != brInfo) {
            TStreamerElement *newElems = (TStreamerElement*)
               newInfo->GetElements()->FindObject(currentStreamerElement->GetName());
            newType = newElems->GetClassPointer()->GetName();
         }
      }
   } else {
      newType = dm->GetTypeName();
   }
   cl = gROOT->GetClass(newType);
   if ( cl ) fCurrentClass = cl;
   return cl;
}

namespace {
   void CleanParentName(TString& parentName, const char* prefix)  {
      if (parentName.Index(prefix)==0) {
         parentName.Remove(0,strlen(prefix));
      }
   }
}

//______________________________________________________________________________
Int_t TBranchElement::GetDataMemberOffsetEx(TClass* par_cl, TString& parentName, Int_t off)
{
   // remove the current data member name

   Ssiz_t pos = parentName.Last('.');
   if (pos>0) {
      // We had a branch name of the style:
      //     [X.]Y.Z
      // and we are looking up 'Y'
      parentName.Remove(pos);
      return GetDataMemberOffset(par_cl,parentName);
   } else {
      // We had a branch name of the style:
      //     [X.]Z
      // and we are looking up 'Z'
      // Because we are missing 'Y' (or more exactly the name of the
      // thing that contains Z, we can only get the offset of Z and
      // then remove the offset Z inside 'Y' (i.e. lOffset)
      if (pos==0&&parentName[0]=='.') parentName.Remove(0,1);
      return GetDataMemberOffset(par_cl,parentName) - off;
   }
}

//______________________________________________________________________________
void TBranchElement::InitializeOffsets()
{
   // Initialize the object-offset 'cache'

   Int_t nbranches = fBranches.GetEntriesFast();
   TClass *clparent = GetParentClass();
   TClass *clm      = fBranchClass;
   if (fType == 31 || fType == 41) {
      if ( fClassName != fParentName ) {
         // We are in the case where we have one or more missing links.
         // This information is realliable here (or so it seems)
         // (In other cases fParentName does not seems to be set correctly in all cases).

         if (clparent != clm) {
            if (!clparent || !clm)    {
               fInitOffsets = kTRUE;
               return;
            }
            const char *ename = fID<0 ? 0 : ((TStreamerElement*)fInfo->GetElems()[fID])->GetName();
            Int_t lOffset    = clm->GetStreamerInfo()->GetOffset(ename); // offset in the local streamerInfo.

            TBranchElement *parent = (TBranchElement*) GetMother()->GetSubBranch(this);
            TString parentDataName( GetName() );
            // remove the TClonesArray main name (if present)
            CleanParentName(parentDataName,fBranchCount->GetName());
            // remove the parent branch name (if present)
            CleanParentName(parentDataName,parent->GetName());

            // fOffset needs to be the offset to be added to the start of the object
            // stored in the collection to find the beginning of the object/class described
            // by the current TStreamerInfo.   The lOffset will then be re-added to this
            // offset by TStreamerInfo::Write/ReadBuffer.
            if (parentDataName[0]=='.') parentDataName.Remove(0,1);
            fOffset = GetDataMemberOffset(clparent,parentDataName) - lOffset;
         }
      }
      if (gDebug > 0 ) {
         printf("fOffset=%d\n",fOffset);
      }
   }
   if ( nbranches > 0 )  {
      Int_t parentID = 0;
      TClass* parentBranchClass = 0;
      TStreamerElement* elem = fID<0 ? 0 : ((TStreamerElement*)fInfo->GetElems()[fID]);
      const char *ename = elem ? elem->GetName() : 0;
      fBranchOffset = new Int_t[nbranches];
      fBranchTypes  = new Bool_t[nbranches];
      for (Int_t i=0; i<nbranches;i++ )  {
         fBranchOffset[i] = 0;
         TBranch *abranch = (TBranch*)fBranches[i];
         fBranchTypes[i]  = abranch->InheritsFrom(TBranchElement::Class());

         //just in case a TBranch had been added to a TBranchElement!
         if ( !fBranchTypes[i] ) {
            continue;
         }
         TBranchElement *branch = (TBranchElement*)abranch;
         Int_t nb2 = branch->GetListOfBranches()->GetEntriesFast();
         Int_t lOffset = 0; // offset in the local streamerInfo.

         lOffset = clm->GetStreamerInfo()->GetOffset(ename);
         Int_t id = branch->GetID();

         if (!clparent) clparent = clm;
         // Test if we are in the case where the class described by 'clparent'
         // did not get its own branch in the tree.  In this case the immediate
         // parent branch will have a different type.

         // First get the immediate parent (i.e a branch which has 'branch' has
         // a direct sub-branch.

         TStreamerInfo *info = branch->GetInfo();
         TBranchElement *parent = this; // = (TBranchElement*) branch->GetMother()->GetSubBranch(branch);
         //assert(parent==this);

         parentID = parent->GetID();
         assert(parentID>=0 || parentID==-2 || parentID==-1);
         // if the ID was negative, the branch would not have been split!
         // -2 = Base class; -1 = (STL-)Collection
         TStreamerInfo *parentInfo = fInfo; // since parent==this parent->GetInfo();
         assert(parentInfo != 0);

         switch(parentID) {
            case -2:
            case -1:
               parentBranchClass = parentInfo->GetClass();
               break;
            default: {
               TStreamerElement *parentElem = (TStreamerElement*)parentInfo->GetElems()[parentID];
               parentBranchClass = parentElem->GetClassPointer();
               break;
            }
         }
         if ( nb2 > 0 )   {
            // The branch has some sub-branches
            if (info) {
               Int_t *leafOffsets = info->GetOffsets();

               TClass *containingClass = info->GetClass();
               if ( ! parentBranchClass->InheritsFrom(containingClass) ) {
                  // We are in the case where there is a missing branch in the hiearchy

                  // Since we do not have a proper hierachy, fObject does NOT point
                  // to object of type 'clparent'.  Instead it points to an object
                  // which 'contains' an object of type 'clparent'.  So the first
                  // order of business is to find the address of the object of type
                  // 'clparent' and then just add 'leafOffsets[id]' (i.e. the offset
                  // of this branch inside the object of type 'clparent'

                  // We need to extract from branch->GetName() the qualified name
                  // of the data member which contains us

                  TString parentDataName = branch->GetName();
                  // remove the main branch name (if present)
                  CleanParentName(parentDataName,branch->GetMother()->GetName());
                  if (leafOffsets) {
                     Int_t offset = GetDataMemberOffsetEx(parentBranchClass, parentDataName, lOffset);
                     fBranchOffset[i] = offset + leafOffsets[id];
                  } else {
                     Error("SetAddress","info=%s, leafOffsets=0",info->GetName());
                  }
               } else {
                  // Case where we have a proper branch hierachy fObject is already correct!
                  if (leafOffsets) {
                     fBranchOffset[i] = leafOffsets[id];
                  } else {
                     Error("SetAddress","info=%s, leafOffsets=0",info->GetName());
                  }
               }
            } else {
               // fInfo==0
               Error("SetAddress","branch=%s, info=0",branch->GetName());
            }
         }
         else {
            Bool_t fixedBaseOffset = kFALSE;
            if ( fType == 1 ) {
               // Offset seems to need correction for TStreamerBases
               std::string myName(GetName());
               Bool_t stripSuffix = kTRUE;
               TBranch* mother = GetMother();
               if (mother == GetSubBranch(this)) {
                  const char* motherName = mother->GetName();
                  if (strlen(motherName) && (motherName[strlen(motherName)-1] == '.')) {
                     stripSuffix = kFALSE;
                     myName.append(".");
                  }
               }
               if (stripSuffix) {
                  size_t pos = myName.find_last_of('.');
                  if (pos == std::string::npos) {
                     myName.clear();
                  } else {
                     myName.erase(pos+1);
                  }
               }
               std::string brName(branch->GetName());
               brName.erase(brName.begin(), brName.begin() + myName.size());
               //const char *name = branch->GetName();
               const char *name = brName.c_str();
               const char *pos = strchr(name, '.');
               if (pos) {
                  size_t idx = pos - name;
                  // Broken branch hierarchy: need to look for offset
                  // in the parents StreamerInfo if the branch represents
                  // a TStreamerBase
                  TClass *pbc = parentBranchClass;
                  TStreamerInfo *pInfo = 0;
                  if (pbc && (pInfo = pbc->GetStreamerInfo()))  {
                     //std::string enam(branch->GetName(), idx);
                     std::string enam(name, idx);
                     fixedBaseOffset = kTRUE;
                     fBranchOffset[i] = pInfo->GetOffset(enam.c_str());
                  }
                  else  {
                     Error("SetAddress","branch=%s, parentInfo==0",branch->GetName());
                  }
               }
            }
            TClass *kidClass = branch->fBranchClass;
            TClass *pClass = branch->GetParentClass();
            if ( !(branch->fType == 31 || branch->fType == 41)
                && pClass && pClass != kidClass) {
               // We need to discover if 'this' represents a direct datamember of the class in parent or
               // if it is an indirect one (with a missing branch in the hierachy)

               assert(parentID>=0);  // if the ID was negative, the branch would not have been split!
               assert(parentInfo != 0);

               if ( ! parentBranchClass->InheritsFrom(kidClass) ) {

                  // We are in the case where there is a missing branch in the hiearchy
                  TString parentDataName( branch->GetName() );
                  const char *ename = branch->fID<0 ? 0 : ((TStreamerElement*)branch->fInfo->GetElems()[branch->fID])->GetName();
                  Int_t lOffset    = kidClass->GetStreamerInfo()->GetOffset(ename); // offset in the local streamerInfo.
                  TString parentName( parent->GetName() );
                  TStreamerElement *parentElem = (TStreamerElement*)parentInfo->GetElems()[parentID];

                  if (parentElem->IsBase()) {
                     TBranchElement *pparent = (TBranchElement*) parent->GetMother()->GetSubBranch(parent);
                     if (pparent != branch->GetMother())  // And not at the 2nd level
                     {
                        TString pattern( Form(".%s",parentElem->GetName()) );
                        if (pattern.Length()<parentName.Length()) {
                           if ( strcmp(parentName.Data()+(parentName.Length()-pattern.Length()),
                              pattern.Data()) == 0 ) {
                                 // The parent branch name contains the name of the base class in it.
                                 // This name is not reproduce in the sub-branches, so we need to
                                 // remove it.
                                 parentName.Remove(parentName.Length()-pattern.Length());
                           }
                        }
                     }
                  }
                  // remove the parent branch name (if present)
                  CleanParentName(parentDataName,parentName);

                  Int_t parentOffset = GetDataMemberOffsetEx(parentBranchClass, parentDataName, lOffset);
                  if (fixedBaseOffset) {
                     // If all the condition above are fullfilled we have already
                     // compensated for (some of) the missing branch(es).
                     parentOffset -= fBranchOffset[i];
                  }
                  // Now transfer parentOffset up to the parent so that it is only
                  // used when recursively setting the addresses.
                  parent->fBranchOffset[i] += parentOffset;
               } else {
                  // Case where we have a proper branch hierachy
                  // fObject is already correct!

                  // nothing to do :)
                  // fprintf(stderr,"section has nothing to do!\n");
               }
            }
         }
      }
   }
   fInitOffsets = kTRUE;
}

//______________________________________________________________________________
void TBranchElement::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*Set address of this branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//

   if (TestBit(kDoNotProcess)) return;
   if (fType < 0) return;

   //special case when called from code generated by TTree::MakeClass
   if (Long_t(add) == -1) {
      fAddress = (char*)add;
      return;
   }
   fReadEntry = -1;

   //build the StreamerInfo if first time for the class
   TClass *cl = fBranchClass;
   if ( !fInfo ) GetInfo();
   Int_t nbranches = fBranches.GetEntriesFast();
   if (gDebug > 0) {
      printf("SetAddress, branch:%s, classname=%s, parent=%s, fID=%d, fType=%d, nbranches=%d, add=%p, fInfo=%s, version=%d => ",
             GetName(),fClassName.Data(),fParentName.Data(),fID,fType,nbranches,add,fInfo->GetName(),fClassVersion);
   }
   fAddress = (char*)add;
   if (fTree->GetMakeClass()) {
      if (fID >= 0) {
         if (!fInfo) {fObject=fAddress; return;}
         fObject = fAddress - fInfo->GetOffsets()[fID];
         return;
      }
   }
   if (fID < 0) {
      if (fAddress) {
         char **ppointer = (char**)fAddress;
         fObject = *ppointer;
         if (!fObject && cl) {
            //Remember if we build an object of an emulated class
            if (!cl->GetClassInfo()) SetBit(kDeleteObject);
            fObject = (char*)cl->New();
            *ppointer = fObject;
         }
      } else {
         //Remember if we build an object of a emulated class
         if (!cl->GetClassInfo()) SetBit(kDeleteObject);
         fObject = (char*)cl->New();
      }
      if (!fAddress) fAddress = (char*)&fObject;
   } else {
      fObject = fAddress;
   }

   // Check whether the container type is still the same
   if (fType ==3) {
      TClass *clm = gROOT->GetClass(fClonesName.Data());
      if (clm) {
         clm->BuildRealData(); //just in case clm derives from an abstract class
         clm->GetStreamerInfo();
      }
      TClass *newType = GetCurrentClass();
      if ( newType && newType != TClonesArray::Class() ) {
         // The data type of the container was changed

         // Let's check if it is a compatible type:
         Bool_t matched = kFALSE;
         if (newType->GetCollectionProxy()) {
            TClass *content = newType->GetCollectionProxy()->GetValueClass();
            if (clm==content) {
               matched = kTRUE;
            } else {
               Warning("SetAddress",
                  "The type of %s was changed from TClonesArray to %s but the content do not match (was %s)!",
                  GetName(),newType->GetName(),fClonesName.Data());
            }
         } else {
            Warning("SetAddress",
               "The type of the %s was changed from TClonesArray to %s but we do not have a TVirtualCollectionProxy for that container type!",
               GetName(),newType->GetName());
         }
         if (matched) {
            // Change from 3/31 to 4/41
            SetType(4);
            SwitchContainer(GetListOfBranches());
            // Set the proxy.
            fSTLtype = TMath::Abs(TClassEdit::IsSTLCont(newType->GetName()));
            fCollProxy = newType->GetCollectionProxy()->Generate();
         } else {
            fAddress = 0;
         }
      }
   }

   //Special case for an STL container.
   if (fType==4) {
      TClass *newType = GetCurrentClass();
      if ( newType && newType != GetCollectionProxy()->GetCollectionClass() ) {

         // Let's check if it is a compatible type:
         TVirtualCollectionProxy *newProxy = newType->GetCollectionProxy();
         TVirtualCollectionProxy *oldProxy = GetCollectionProxy();
         if (newProxy && oldProxy->GetValueClass() == newProxy->GetValueClass()
            && ( (oldProxy->GetValueClass() ==0 &&
                  oldProxy->GetType() == newProxy->GetType()) ||
                 (oldProxy->GetValueClass() &&
                  oldProxy->HasPointers() == newProxy->HasPointers()) ) ) {

            if ( fSTLtype==TClassEdit::kNotSTL ) {
               fSTLtype = TMath::Abs(TClassEdit::IsSTLCont(newType->GetName()));
            }
            fCollProxy = newType->GetCollectionProxy()->Generate();
         } else {
            // The new collection and the old collection are not compatible,
            // we can not use the new collection to read the data.
            // Actually if could check whether the new collection is a
            // compatible ROOT collection.
            if (  newType == TClonesArray::Class() &&
                  ( oldProxy->GetValueClass() && !oldProxy->HasPointers()
                   && oldProxy->GetValueClass()->InheritsFrom(TObject::Class()))) {
               // We can not insure that the TClonesArray is set for the
               // proper class ( oldProxy->GetValueClass() ), so we assume that
               // the transformation was done properly by the class designer.

               // Change from 4/41 to 3/31
               SetType(3);
               SwitchContainer(GetListOfBranches());
               // Reset the proxy.
               fSTLtype = kNone;
               switch(fStreamerType) {
                  case TStreamerInfo::kAny:
                  case TStreamerInfo::kSTL:
                     fStreamerType = TStreamerInfo::kObject;
                     break;
                  case TStreamerInfo::kAnyp:
                  case TStreamerInfo::kSTLp:
                     fStreamerType = TStreamerInfo::kObjectp;
                     break;
                  case TStreamerInfo::kAnyP:
                     fStreamerType = TStreamerInfo::kObjectP;
                     break;
               }
               fClonesName = oldProxy->GetValueClass()->GetName();
               delete fCollProxy; fCollProxy = 0;
               TClass *clm = gROOT->GetClass(fClonesName);
               if (clm) {
                  clm->BuildRealData(); //just in case clm derives from an abstract class
                  clm->GetStreamerInfo();
               }
            } else {
               fAddress = 0;
            }
         }
      }
   }

   //special case for a TClonesArray when address is not yet set
   //we must create the clonesarray first
   if (fType==3) {
      if (fAddress) {
         if (fStreamerType==61) {
            // Case of an embedded ClonesArray
            fObject = fAddress;
            // Check if it has already been properly build.
            TClonesArray *clones = (TClonesArray*)fObject;
            if (clones->GetClass()==0) {
               new (fObject) TClonesArray(fClonesName.Data());
            }
         } else {
            TClonesArray **ppointer = (TClonesArray**)fAddress;
            if (!*ppointer) *ppointer = new TClonesArray(fClonesName.Data());
            fObject = (char*)*ppointer;
         }
         if (!fObject) fAddress = 0;
      }
      if (!fAddress) {
         //SetBit(kDeleteObject);
         fObject = (char*)new TClonesArray(fClonesName.Data());
         fAddress = (char*)&fObject;
      }

   } else if (fType==4) {
      TVirtualCollectionProxy* proxy = GetCollectionProxy(); // initialize fCollProxy
      if (fAddress) {
         if (fStreamerType==61 ||
             fStreamerType==TStreamerInfo::kAny ||
             fStreamerType==TStreamerInfo::kSTL) {
            // Case of an embedded container?
            fObject = fAddress;
         } else {
            void **ppointer = (void**)fAddress;
            if (!*ppointer) *ppointer = proxy->New();
            fObject = (char*)*ppointer;
         }
         if (!fObject) fAddress = 0;
      }
      if (!fAddress) {
         //SetBit(kDeleteObject);
         fObject  = (char*)proxy->New();
         fAddress = (char*)&fObject;
      }
   } else if (fType==41) {
      GetCollectionProxy(); // initialize fCollProxy
   }

   if (gDebug > 0 ) {
      printf("fAddress=%p, fObject=%p, \n",fAddress,fObject);
   }

   if ( !fInfo ) return;
   if ( !fInitOffsets ) InitializeOffsets();

   if (fType == 31 || fType == 41) {
      return;
   }
   for (Int_t i=0;i<nbranches;i++)  {
      TBranch *abranch = (TBranch*)fBranches[i];
      abranch->SetAddress(fObject + fBranchOffset[i]);
   }
}

//______________________________________________________________________________
void TBranchElement::SetAutoDelete(Bool_t autodel)
{
//*-*-*-*-*-*-*-*Set the AutoDelete bit
//*-*            ====================
//  This function can be used to instruct Root in TBranchElement::ReadBasket
//  to not delete the object referenced by a branchelement before reading a
//  new entry. By default, the object is deleted.
//  If autodel is kTRUE, this existing object will be deleted, a new object
//    created by the default constructor, then object->Streamer called.
//  If autodel is kFALSE, the existing object is not deleted. Root assumes
//    that the user is taking care of deleting any internal object or array
//    This can be done in Streamer itself.
//  If this branch has sub-branches, the function sets autodel for these
//  branches as well.
//  We STRONGLY suggest to activate this option by default when you create
//  the top level branch. This will make the read phase more efficient
//  because it minimizes the numbers of new/delete operations.
//  Once this option has been set and the Tree is written to a file, it is
//  not necessary to specify the option again when reading, unless you
//  want to set the opposite mode.
//

   TBranch::SetAutoDelete(autodel);
}

//______________________________________________________________________________
void TBranchElement::SetBasketSize(Int_t buffsize)
{
// Reset basket size for all subbranches of this branchelement

   TBranch::SetBasketSize(buffsize);

   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->SetBasketSize(fBasketSize);
   }
}

//______________________________________________________________________________
void TBranchElement::SetBranchCount(TBranchElement *bre)
{
// Set the branch count for this branch
   fBranchCount = bre;
   TLeafElement *lfc  = (TLeafElement *)bre->GetListOfLeaves()->At(0);
   TLeafElement *leaf = (TLeafElement *)GetListOfLeaves()->At(0);
   if (lfc && leaf) leaf->SetLeafCount(lfc);
}

//______________________________________________________________________________
void TBranchElement::Streamer(TBuffer &R__b)
{
   // Stream an object of class TBranchElement.

   if (R__b.IsReading()) {
      TBranchElement::Class()->ReadBuffer(R__b, this);
      fParentClass.SetName(fParentName);
      fBranchClass.SetName(fClassName);

      // Fixup a case where the TLeafElement was missing
      if (fType==0 && fLeaves.GetEntriesFast()==0) {
         TLeaf *leaf     = new TLeafElement(GetTitle(),fID, fStreamerType);
         leaf->SetTitle(GetTitle());
         leaf->SetBranch(this);
         fNleaves = 1;
         fLeaves.Add(leaf);
         fTree->GetListOfLeaves()->Add(leaf);
      }
   } else {
      TDirectory *dirsav = fDirectory;
      fDirectory = 0;  // to avoid recursive calls

      TBranchElement::Class()->WriteBuffer(R__b, this);

      // make sure that all TStreamerInfo objects referenced by
      // this class are written to the file
      if (fInfo) fInfo->ForceWriteInfo((TFile *)R__b.GetParent(), kTRUE);

      // if branch is in a separate file save this branch
      // as an independent key
      if (!dirsav) return;
      if (!dirsav->IsWritable()) {fDirectory = dirsav; return;}
      TDirectory *pdirectory = fTree->GetDirectory();
      if (!pdirectory) {fDirectory = dirsav; return;}
      const char *treeFileName = pdirectory->GetFile()->GetName();
      TBranch *mother = GetMother();
      const char *motherFileName = treeFileName;
      if (mother && mother != this) {
         motherFileName = mother->GetFileName();
      }
      if (fFileName.Length() > 0 && strcmp(motherFileName,fFileName.Data())) {
         TDirectory *cursav = gDirectory;
         dirsav->cd();
         Write();
         cursav->cd();
      }
      fDirectory = dirsav;
   }
}

//______________________________________________________________________________
Int_t TBranchElement::Unroll(const char *name, TClass *cltop, TClass *cl,Int_t basketsize, Int_t splitlevel, Int_t btype)
{
// unroll base classes and loop on all elements of class cl

   if (cl == TObject::Class() && cltop->CanIgnoreTObjectStreamer()) return 0;
   Bool_t optim = TStreamerInfo::CanOptimize();
   if (splitlevel > 0) TStreamerInfo::Optimize(kFALSE);
   TStreamerInfo *info = fTree->BuildStreamerInfo(cl);
   TStreamerInfo::Optimize(optim);
   if (!info) return 0;
   TClass *clbase;
   Int_t ndata = info->GetNdata();
   ULong_t *elems = info->GetElems();
   TStreamerElement *elem;
   TBranchElement *branch;
   char branchname[kMaxLen];
   Int_t jd = 0;
   Int_t unroll = 0;
   if (ndata==1 && cl->GetCollectionProxy() &&
      strcmp(((TStreamerElement*)elems[0])->GetName(),"This")==0) {
      // This streamerInfo only refers to the collection itself
      return 1;
   }
   for (Int_t i=0;i<ndata;i++) {
      elem = (TStreamerElement*)elems[i];
      Int_t offset = elem->GetOffset();
      char *oldPointer = fBranchPointer;
      if (gDebug > 0) printf("Unroll name=%s, cltop=%s, cl=%s, i=%d, elem=%s, offset=%d, splitlevel=%d, fBranchPointer=%lx, btype=%d \n",name,cltop->GetName(),cl->GetName(),i,elem->GetName(),elem->GetOffset(),splitlevel,(Long_t)fBranchPointer,btype);
      if (elem->IsA() == TStreamerBase::Class()) {
         clbase = gROOT->GetClass(elem->GetName());

         if (clbase->Property() & kIsAbstract) {
            if (cl->InheritsFrom("TCollection")) unroll = -1;
         }
         if (unroll < 0 && (btype != 31 || btype != 41)) return -1;

         if (gDebug > 0) printf("Unrolling base class, cltop=%s, clbase=%s\n",cltop->GetName(),clbase->GetName());
         if (btype==31 || btype==41) {
            fBranchPointer += offset;
            unroll = Unroll(name,cltop,clbase,basketsize,splitlevel-1,btype);
            fBranchPointer = oldPointer;
            if (unroll < 0) {
               if (strlen(name)) sprintf(branchname,"%s.%s",name,elem->GetFullName());
               else              sprintf(branchname,"%s",elem->GetFullName());
               branch = new TBranchElement(branchname,info,jd,0,basketsize,0,btype);
               branch->SetParentClass(cltop);
               fBranches.Add(branch);
            }
         } else if (clbase->GetListOfRealData()->GetSize()!=0) {

            // We do not create a branch for an empty base class:
            char *pointer = fBranchPointer + offset;
            if (strlen(name)) {
               sprintf(branchname,"%s.%s",name,elem->GetFullName());
               // First claim that we have the short name (to fool the children)
               branch = new TBranchElement(name,info,jd,pointer,basketsize,splitlevel,btype);
               // Then reset it to the proper name
               branch->SetName(branchname);
               branch->SetTitle(branchname);
            } else {
               sprintf(branchname,"%s",elem->GetFullName());
               branch = new TBranchElement(branchname,info,jd,pointer,basketsize,splitlevel,btype);
            }
            // branch = new TBranchElement(branchname,info,jd,pointer,basketsize,splitlevel-1,btype);
            branch->SetParentClass(cltop);
            fBranches.Add(branch);
            if (0 && unroll < 0) {
               branch = new TBranchElement(branchname,info,jd,0,basketsize,0,btype);
               branch->SetParentClass(cltop);
               fBranches.Add(branch);
            }
         }
      } else {
         if (strlen(name)) sprintf(branchname,"%s.%s",name,elem->GetFullName());
         else              sprintf(branchname,"%s",elem->GetFullName());
         if (  splitlevel > 1 &&
               ( elem->IsA() == TStreamerObject::Class() || elem->IsA() == TStreamerObjectAny::Class() ) ) {

            clbase = gROOT->GetClass(elem->GetTypeName());
            if (clbase->Property() & kIsAbstract) return -1;

            if (gDebug > 0) printf("Unrolling object class, cltop=%s, clbase=%s\n",cltop->GetName(),clbase->GetName());
            fBranchPointer += offset;
            if (elem->CannotSplit()) {

               unroll = -1;

            } else if  (clbase->InheritsFrom(TClonesArray::Class())) {

               branch = new TBranchElement(branchname,info,jd,fBranchPointer ,basketsize,splitlevel-1,btype);
               branch->SetParentClass(cltop);
               fBranches.Add(branch);
               unroll = 0;

            } else {

               unroll = Unroll(branchname,cltop,clbase,basketsize,splitlevel-1,btype);
            }
            fBranchPointer = oldPointer;
            if (unroll < 0) {
               char *pointer = fBranchPointer + offset;
               branch = new TBranchElement(branchname,info,jd,pointer,basketsize,0,btype);
               branch->SetParentClass(cltop);
               fBranches.Add(branch);
            }
         } else if (elem->IsA() == TStreamerSTL::Class() && !elem->IsaPointer()) {
            // here all STL classes are handled
            Int_t subSplitlevel = splitlevel-1;
            if (btype == 31 || btype == 41 || elem->CannotSplit()) {
               subSplitlevel = 0;
            }
            char* pointer = fBranchPointer + offset;
            branch = new TBranchElement(branchname,info,jd,/* 0 */ pointer ,basketsize,subSplitlevel,btype);
            branch->SetParentClass(cltop);
            fBranches.Add(branch);

         } else {
            //fBranchPointer may be null in case of a TClonesArray inside another TClonesArray
            if ((btype != 31 && btype != 41) && fBranchPointer &&
                ( elem->GetClassPointer() == TClonesArray::Class()
                  || (elem->IsA() == TStreamerSTL::Class() && !elem->CannotSplit())
                  )
                ) {
               //process case of a TClonesArray in a derived class
               char *pointer = fBranchPointer + offset;
               branch = new TBranchElement(branchname,info,jd,pointer,basketsize,splitlevel-1,btype);
            } else {
               branch = new TBranchElement(branchname,info,jd,0,basketsize,0,btype);
               branch->SetType(btype);
            }
            branch->SetParentClass(cltop);
            fBranches.Add(branch);
         }
      }
      jd++;
   }
   return 1;
}

//______________________________________________________________________________
TVirtualCollectionProxy *TBranchElement::GetCollectionProxy()
{
   // Return the TVirtualCollectionProxy describing the branch
   // content, if any.

   if (fCollProxy) return fCollProxy;

   TBranchElement *thiscast = const_cast<TBranchElement*>(this);
   if (fType==4 ) {

      const char *ty;
      if (fID>=0) ty = ((TStreamerElement*)thiscast->GetInfo()->GetElems()[fID])->GetTypeName();
      else ty = fClassName.Data();
      TClass *cl = gROOT->GetClass(ty);
      fCollProxy = cl->GetCollectionProxy()->Generate();
      fSTLtype   = TClassEdit::IsSTLCont(ty);
      if ( fSTLtype<0 ) fSTLtype = -fSTLtype;
   } else if(fType==41) {
      thiscast->fCollProxy = fBranchCount->fCollProxy;
   }
   return fCollProxy;
}
