// @(#)root/tree:$Name:  $:$Id: TBranchElement.cxx,v 1.88 2002/04/22 19:02:31 brun Exp $
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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
#include "TClonesArray.h"
#include "TTree.h"
#include "TBasket.h"
#include "TLeafElement.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TRealData.h"
#include "TDataType.h"
#include "TDataMember.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TBrowser.h"
#include "TFolder.h"
#include "Api.h"

const Int_t kMaxLen = 1024;
R__EXTERN  TTree *gTree;

ClassImp(TBranchElement)

//______________________________________________________________________________
TBranchElement::TBranchElement(): TBranch()
{
//*-*-*-*-*-*Default constructor for BranchElement*-*-*-*-*-*-*-*-*-*
//*-*        ====================================

   fNleaves = 1;
   fInfo = 0;
   fBranchCount = 0;
   fBranchCount2 = 0;
   fObject = 0;
   fMaximum = 0;
   fBranchPointer = 0;
   fNdata = 1;
}


//______________________________________________________________________________
TBranchElement::TBranchElement(const char *bname, TStreamerInfo *sinfo, Int_t id, char *pointer, Int_t basketsize, Int_t splitlevel, Int_t btype)
    :TBranch()
{
// Create a BranchElement
//
// If splitlevel > 0 this branch in turn is split into sub branches

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
   fID           = id;
   fStreamerType = -1;
   fType         = 0;
   fBranchCount  = 0;
   fBranchCount2 = 0;
   fObject       = 0;
   fBranchPointer= 0;
   fNdata        = 1;
   fClassVersion = cl->GetClassVersion();
   fTree         = gTree;
   fMaximum      = 0;
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

   fDirectory  = fTree->GetDirectory();
   fFileName   = "";
   fClassName = sinfo->GetName();
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
   fBasketEntry    = new Int_t[fMaxBaskets];
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketSeek     = new Seek_t[fMaxBaskets];

   fBasketEntry[0] = fEntryNumber;
   fBasketBytes[0] = 0;

   // Create a basket for the terminal branch
   TBasket *basket = new TBasket(name,fTree->GetName(),this);
   fBaskets.Add(basket);

   // save pointer (if non null). Will be used in Unroll in case we find
   // a TClonesArray in a derived class.
   if (pointer) fBranchPointer = pointer;
   
   // create sub branches if requested by splitlevel

   if (splitlevel > 0) {
      TClass *clm;
      if (element->CannotSplit()) {
         //printf("element: %s/%s will not be split\n",element->GetName(),element->GetTitle());
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

      } else if (!strcmp(element->GetTypeName(),"TClonesArray") || !strcmp(element->GetTypeName(),"TClonesArray*")) {
         Bool_t ispointer = !strcmp(element->GetTypeName(),"TClonesArray*");
         TClonesArray *clones;
         if (ispointer) {
            char **ppointer = (char**)(pointer);
            clones = (TClonesArray*)(*ppointer);
         } else {
            clones = (TClonesArray*)pointer;
         }
         basket->DeleteEntryOffset(); //entryoffset not required for the clonesarray counter
         fEntryOffsetLen = 0;
         clm = clones->GetClass();
         if (!clm) return;
         // ===> Create a leafcount
         TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
         leaf->SetBranch(this);
         fNleaves = 1;
         fLeaves.Add(leaf);
         fTree->GetListOfLeaves()->Add(leaf);
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

      } else if (!strchr(element->GetTypeName(),'*') && (fStreamerType == TStreamerInfo::kObject || fStreamerType == TStreamerInfo::kAny)) {
         // ===> create sub branches for members that are classes
         fType = 2;
         clm = gROOT->GetClass(element->GetTypeName());
         if (Unroll(name,clm,clm,basketsize,splitlevel,0) >= 0) return;
      } else if (strstr(element->GetTypeName(),"vector<")) {
         // ===> create sub branches for each data member of a STL vector
         //      if it is a vector of class objects.
         //      STL vectors like vector<float> are not split
         fType = 0;
         char classname[kMaxLen];
         strcpy(classname,element->GetTypeName()+7);
         char *star = strchr(classname,'>');
         *star = 0;
         clm = gROOT->GetClass(classname);
         if (clm) {
            TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
            leaf->SetBranch(this);
            fNleaves = 1;
            fLeaves.Add(leaf);
            fTree->GetListOfLeaves()->Add(leaf);
            // Create a basket for the leafcount
            TBasket *basket = new TBasket(name,fTree->GetName(),this);
            fBaskets.Add(basket);
            // ===> create sub branches for each data member of the class
            fType = 4;
            char branchname[kMaxLen];
            sprintf(branchname,"%s_",name);
            SetTitle(branchname);
            Unroll(name,clm,clm,basketsize,splitlevel,41);
            TStreamerSTL *stl = (TStreamerSTL*)element;
            if (stl->GetSTLtype() == 1 && stl->GetCtype() == 61) {
               printf("will split this STL vector: %s of %s\n",name,classname);
            }
            if (stl->GetSTLtype() == 41 && stl->GetCtype() == 61) {
               printf("will split this pointer to STL vector: %s of %s\n",name,classname);
            }
            return;
         }
      }
   }

   TLeaf *leaf     = new TLeafElement(GetTitle(),fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   leaf->SetBranch(this);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);
   if (brcount) SetBranchCount(brcount); 
}

//______________________________________________________________________________
TBranchElement::TBranchElement(const char *bname, TClonesArray *clones, Int_t basketsize, Int_t splitlevel, Int_t compress)
    :TBranch()
{
// Create a BranchElement
//
// If splitlevel > 0 this branch in turn is split into sub branches

   char name[kMaxLen];
   strcpy(name,bname);
   fSplitLevel   = splitlevel;
   fInfo         = TClonesArray::Class()->GetStreamerInfo();
   fID           = 0;
   fStreamerType = -1;
   fType         = 0;
   fClassVersion = TClonesArray::Class()->GetClassVersion();
   fBranchCount  = 0;
   fBranchCount2 = 0;
   fObject       = 0;
   fBranchPointer= 0;
   fMaximum      = 0;

   fTree       = gTree;
   fDirectory  = fTree->GetDirectory();
   fFileName   = "";

   SetName(name);
   SetTitle(name);
   fClassName = fInfo->GetName();
   fCompress = compress;
   if (compress == -1 && fTree->GetDirectory()) {
      TFile *bfile = fTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }

   if (basketsize < 100) basketsize = 100;
   fBasketSize     = basketsize;
   fBasketEntry    = new Int_t[fMaxBaskets];
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketSeek     = new Seek_t[fMaxBaskets];

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
      char branchname[kMaxLen];
      sprintf(branchname,"%s_",name);
      SetTitle(branchname);
      leaf->SetName(branchname);
      leaf->SetTitle(branchname);
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
TBranchElement::~TBranchElement()
{
//*-*-*-*-*-*Default destructor for a BranchElement*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =====================================

   fBranches.Delete();
   
   //SetAddress may have allocated an object. Must delete it
   if (TestBit(kDeleteObject)) {
      if (fObject) {
         //TObject *obj = (TObject*)fObject;
         // objects of fake classes allocated in SetAddress should be deleted
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
   char *name = new char[1000];
   sprintf(name,"%s.%s",GetName(),subname);
   TBranch *branch = new TBranch(name,address,leaflist,bufsize);
   if (branch->IsZombie()) {
      delete branch;
      delete [] name;
      return 0;
   }
   TLeaf *leaf;
   TIter next(branch->GetListOfLeaves());
   while ((leaf = (TLeaf*)next())) {
      sprintf(name,"%s.%s",GetName(),leaf->GetName());
      leaf->SetName(name);
   }
   delete [] name;
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
   char *curname = new char[1000];
   char occur[20];
   while ((obj=next())) {
      sprintf(curname,"%s/%s",foldername,obj->GetName());
      if (obj->IsA() == TFolder::Class()) {
        Branch(curname, bufsize, splitlevel-1);
      } else {
        void *add = (void*)folder->GetListOfFolders()->GetObjectRef(obj);
        for (Int_t i=0;i<1000;i++) {
           if (curname[i] == 0) break;
           if (curname[i] == '/') curname[i] = '.';
        }
        Int_t noccur = folder->Occurence(obj);
        if (noccur > 0) {
           sprintf(occur,"_%d",noccur);
           strcat(curname,occur);
        }
        TBranchElement *br;
        br = (TBranchElement*)Branch(curname,obj->ClassName(), add, bufsize, splitlevel-1);
        br->SetBranchFolder();
      }
   }
   delete [] curname;
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
   char *name = new char[1000];
   sprintf(name,"%s.%s",GetName(),subname);
   if (cl == TClonesArray::Class()) {
      TClonesArray *clones = (TClonesArray *)objadd;
      if (!clones) {
         Error("Branch","Pointer to TClonesArray is null");
         delete [] name;
         return 0;
      }
      if (!clones->GetClass()) {
         Error("Branch","TClonesArray with no class defined in branch: %s",name);
         delete [] name;
         return 0;
      }
      G__ClassInfo *classinfo = clones->GetClass()->GetClassInfo();
      if (!classinfo) {
         Error("Bronch","TClonesArray with no dictionary defined in branch: %s",name);
         return 0;
      }
      if (splitlevel > 0) {
         if (classinfo->RootFlag() & 1)
            Warning("Branch","Using split mode on a class: %s with a custom Streamer",clones->GetClass()->GetName());
      } else {
         if (classinfo->RootFlag() & 1) clones->BypassStreamer(kFALSE);
         TBranchObject *branch = new TBranchObject(name,classname,add,bufsize,0);
         fBranches.Add(branch);
         delete [] name;
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
      TBranchObject *branch = new TBranchObject(name,classname,add,bufsize,0);
      fBranches.Add(branch);
      delete [] name;
      return branch;
   }

   //hopefully normal case
   Bool_t delobj = kFALSE;
   //====> special case of TClonesArray
   if(cl == TClonesArray::Class()) {
      TBranchElement *branch = new TBranchElement(name,(TClonesArray*)objadd,bufsize,splitlevel);
      fBranches.Add(branch);
      branch->SetAddress(add);
      delete [] name;
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
   char *dot = (char*)strchr(name,'.');
   Int_t nch = strlen(name);
   Bool_t dotlast = kFALSE;
   if (nch && name[nch-1] == '.') dotlast = kTRUE;
   TBranchElement *branch = new TBranchElement(name,sinfo,id,objadd,bufsize,splitlevel);
   fBranches.Add(branch);
   if (splitlevel > 0) {

      // Loop on all public data members of the class and its base classes
      TObjArray *blist = branch->GetListOfBranches();
      TIter next(sinfo->GetElements());
      TStreamerElement *element;
      id = 0;
      char *bname = new char[1000];
      while ((element = (TStreamerElement*)next())) {
         char *pointer = (char*)objadd + element->GetOffset();
         Bool_t isBase = element->IsA() == TStreamerBase::Class();
         if (isBase) {
            TClass *clbase = element->GetClassPointer();
            if (clbase == TObject::Class() && cl->CanIgnoreTObjectStreamer()) continue;
         }
         if (dot) {
            if (dotlast) {
               sprintf(bname,"%s%s",name,element->GetFullName());
            } else {
               if (isBase) sprintf(bname,"%s",name);
               else        sprintf(bname,"%s.%s",name,element->GetFullName());
            }
         } else {
            sprintf(bname,"%s",element->GetFullName());
         }
         TBranchElement *bre = new TBranchElement(bname,sinfo,id,pointer,bufsize,splitlevel-1);
         blist->Add(bre);
         id++;
      }
      delete [] bname;
   }
   delete [] name;

   branch->SetAddress(add);

   if (delobj) {delete objadd; *ppointer=0;}
   return branch;
}


//______________________________________________________________________________
void TBranchElement::Browse(TBrowser *b)
{
   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches > 0) {
      fBranches.Browse(b);
   } else {
      // Get the name and strip any extra brackets
      // in order to get the full arrays.
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

      GetTree()->Draw(name);
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
      bre->SetType(31);
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
      if (stype > 40 && stype < 55) {
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

   Int_t nbytes = 0;
   Int_t nbranches = fBranches.GetEntriesFast();
   // update addresses if top level branch
   if (fID < 0) {
      if (!fAddress) {
         Error("Fill","attempt to fill branch %s while addresss is not set",GetName());
         return 0;
      }
      void *add1 = fObject;
      void **add2 = (void**)fAddress;
      if (add1 != *add2) {
         SetAddress(fAddress);
      }
   
   }
   if (nbranches) {
      if (fType == 3)  nbytes += TBranch::Fill();  //TClonesArray counter
      else             fEntries++;
      for (Int_t i=0;i<nbranches;i++)  {
         TBranchElement *branch = (TBranchElement*)fBranches[i];
         if (!branch->TestBit(kDoNotProcess)) nbytes += branch->Fill();
      }
   } else {
      if (!TestBit(kDoNotProcess)) nbytes += TBranch::Fill();
   }
   if (fTree->Debug() > 0) {
      Int_t entry = (Int_t)fEntries;
      if (entry >= fTree->GetDebugMin() && entry <= fTree->GetDebugMax()) {
         printf("Fill: %d, branch=%s, nbytes=%d\n",entry,GetName(),nbytes);
      }
   }
   return nbytes;
}

//______________________________________________________________________________
void TBranchElement::FillLeaves(TBuffer &b)
{
//  Fill buffers of this branch

  if (!fObject) return;
  if (fType <= 2 && TestBit(kBranchObject)) b.MapObject((TObject*)fObject);

  if (fType == 4) {           // STL vector/list of objects
     //printf ("STL split mode not yet implemented\n");
  } else if (fType == 41) {   // sub branch of an STL class
    //char **ppointer = (char**)fAddress;
  } else if (fType == 3) {   //top level branch of a TClonesArray
    TClonesArray *clones = (TClonesArray*)fObject;
    if (!clones) return;
    Int_t n = clones->GetEntriesFast();
    if (n > fMaximum) fMaximum = n;
    b << n;
  } else if (fType == 31) {   // sub branch of a TClonesArray
    TClonesArray *clones = (TClonesArray*)fObject;
    if (!clones) return;
    Int_t n = clones->GetEntriesFast();
    fInfo->WriteBufferClones(b,clones,n,fID,fOffset);
  } else if (fType <= 2) {
    Int_t n = fInfo->WriteBuffer(b,fObject,fID);
    if (fStreamerType == 6) {
       if (n > fMaximum) fMaximum = n;
    }
  }
}

//______________________________________________________________________________
Int_t TBranchElement::GetEntry(Int_t entry, Int_t getall)
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

   // if branch address is not yet set, must set all addresses starting
   // with the top level parent branch
   if (fAddress == 0 && fTree->GetMakeClass() == 0) {
      TBranchElement *mother = GetMother();
      TClass *cl = gROOT->GetClass(mother->GetClassName());
      if (fInfo && fInfo->GetOffsets()) fInfo->BuildOld();
      if (!mother || !cl) return 0;
      if (!mother->GetAddress()) mother->SetAddress(0);
   }

   if (nbranches) {
      //branch has daughters
      //one must always read the branch counter. 
      //In the case when one reads consecutively twice the same entry,
      //the user may have cleared the TClonesArray between the 2 GetEntry
      if (fType == 3) nbytes += TBranch::GetEntry(entry, getall);

      Int_t nb;
      for (Int_t i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches[i];
         nb  = branch->GetEntry(entry, getall);
         if (nb < 0) return nb;
         nbytes += nb;
      }
   } else {
      //terminal branch
      if (fBranchCount && fBranchCount->GetReadEntry() != entry) nbytes += fBranchCount->TBranch::GetEntry(entry,getall);
      nbytes += TBranch::GetEntry(entry, getall);
   }

   if (fTree->Debug() > 0) {
      if (entry >= fTree->GetDebugMin() && entry <= fTree->GetDebugMax()) {
         printf("GetEntry: %d, branch=%s, nbytes=%d\n",entry,GetName(),nbytes);
      }
   }
   return nbytes;
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
      return fInfo;
   }
   TClass *cl = gROOT->GetClass(fClassName.Data());
   if (cl) {
      TStreamerInfo::Optimize(kFALSE);
      if (cl == TClonesArray::Class()) fClassVersion = TClonesArray::Class()->GetClassVersion();
      fInfo = cl->GetStreamerInfo(fClassVersion);
      if (fInfo && !fInfo->GetOffsets()) {
         fInfo->Compile();
      }
      TStreamerInfo::Optimize(optim);
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
TBranchElement *TBranchElement::GetMother() const
{
// Get top level branch parent of this branch
// A top level branch has its fID negative.

   TIter next(fTree->GetListOfBranches());
   TBranch *branch;
   TBranchElement *bre, *br;
   while ((branch=(TBranch*)next())) {
      if (branch->IsA() != TBranchElement::Class()) continue;
      bre = (TBranchElement*)branch;
      br = bre->GetSubBranch(this);
      if (br) return bre;
   }
   return 0;
}

//______________________________________________________________________________
TBranchElement *TBranchElement::GetSubBranch(const TBranchElement *br) const
{
// recursively find branch br in the list of branches of this branch.
// return null if br is not in this branch hierarchy.

   if (br == this) return (TBranchElement*)this;
   TIter next(((TBranchElement*)this)->GetListOfBranches());
   TBranchElement *branch, *br2;
   while ((branch = (TBranchElement*)next())) {
      br2 = branch->GetSubBranch(br);
      if (br2) return br2;
   }
   return 0;
}

//______________________________________________________________________________
const char *TBranchElement::GetTypeName() const
{
   // return type name of element in the branch

   if (fType == 3) {
      return "Int_t";
   }
   if (fStreamerType <=0 || fStreamerType >= 60) return fClassName.Data();
   const char *types[16] = {"","Char_t","Short_t","Int_t","Long_t","Float_t",
      "Int_t","","Double_t","","","UChar_t","UShort_t","UInt_t","ULong_t","UInt_t"};
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
     if (fType == 3) {    //top level branch of a TClonesArray
       return (Double_t)fNdata;
     } else if (fType == 31) {    // sub branch of a TClonesArray
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
   } else {
      return fInfo->GetValue(fObject,fID,j,-1);
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
     } else if (fType == 31) {    // sub branch of a TClonesArray
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
   } else {
      //return fInfo->GetValue(fObject,fID,j,-1);
      if (!fInfo) return 0;
      char **val = (char**)(fObject+fInfo->GetOffsets()[fID]);
      return *val;
   }
}

//______________________________________________________________________________
Bool_t TBranchElement::IsFolder() const
{
//*-*-*-*-*Return TRUE if more than one leaf, FALSE otherwise*-*
//*-*      ==================================================

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches >= 1) return kTRUE;
   else                return kFALSE;
}

//______________________________________________________________________________
void TBranchElement::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print TBranch parameters*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   Int_t nbranches = fBranches.GetEntriesFast();
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
void TBranchElement::PrintValue(Int_t len) const
{
// Prints leaf value
 
  if (fTree->GetMakeClass()) {
     if (!fAddress) return;
     if (fType == 3) {    //top level branch of a TClonesArray
       printf(" %-15s = %d\n",GetName(),fNdata);
       return;
     } else if (fType == 31) {    // sub branch of a TClonesArray
       Int_t n = TMath::Min(10,fNdata);
       Int_t atype = fStreamerType+20;
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
       fInfo->PrintValue(GetName(),fAddress,atype,n);
       return;
     } else if (fType <= 2) {     // branch in split mode
       if (fStreamerType > 40 && fStreamerType < 55) {
          Int_t atype = fStreamerType - 20;
          Int_t n = (Int_t)((TBranchElement*)fBranchCount)->GetValue(0,0);
          fInfo->PrintValue(GetName(),fAddress,atype,n);
       } else {
          fInfo->PrintValue(GetName(),fObject,fID,-1);
       }
       return;
     }
     return;
  }
   if (fType == 3) {
      printf(" %-15s = %d\n",GetName(),fNdata);
   } else if (fType == 31) {
      TClonesArray *clones = (TClonesArray*)fObject;
      fInfo->PrintValueClones(GetName(),clones,fID,fOffset);
   } else {
      fInfo->PrintValue(GetName(),fObject,fID,-1);
   }
}

//______________________________________________________________________________
void TBranchElement::ReadLeaves(TBuffer &b)
{
// Read buffers for this branch

  if (fTree->GetMakeClass()) {
     if (fType == 3) {    //top level branch of a TClonesArray
       Int_t *n = (Int_t*)fAddress;
       b >> n[0];
       fNdata = n[0];
       return;
     } else if (fType == 31) {    // sub branch of a TClonesArray
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
                case  1:  {length = ((Char_t*)len_where)[k]; break;}
                case  2:  {length = ((Short_t*) len_where)[k]; break;}
                case  3:  {length = ((Int_t*)   len_where)[k]; break;}
                case  4:  {length = ((Long_t*)  len_where)[k]; break;}
                   //case  5:  {length = ((Float_t*) len_where)[k]; break;}
                case  6:  {length = ((Int_t*)   len_where)[k]; break;}
                   //case  8:  {length = ((Double_t*)len_where)[k]; break;}
                case 11:  {length = ((UChar_t*) len_where)[k]; break;}
                case 12:  {length = ((UShort_t*)len_where)[k]; break;}
                case 13:  {length = ((UInt_t*)  len_where)[k]; break;}
                case 14:  {length = ((ULong_t*) len_where)[k]; break;}
                default: continue;
             }
             if (length<=0) continue;
             b >> isArray; 
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
          case  1:  {b.ReadFastArray((Char_t*)  fAddress, n); break;}
          case  2:  {b.ReadFastArray((Short_t*) fAddress, n); break;}
          case  3:  {b.ReadFastArray((Int_t*)   fAddress, n); break;}
          case  4:  {b.ReadFastArray((Long_t*)  fAddress, n); break;}
          case  5:  {b.ReadFastArray((Float_t*) fAddress, n); break;}
          case  6:  {b.ReadFastArray((Int_t*)   fAddress, n); break;}
          case  8:  {b.ReadFastArray((Double_t*)fAddress, n); break;}
          case 11:  {b.ReadFastArray((UChar_t*) fAddress, n); break;}
          case 12:  {b.ReadFastArray((UShort_t*)fAddress, n); break;}
          case 13:  {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
          case 14:  {b.ReadFastArray((ULong_t*) fAddress, n); break;}
          case 15:  {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
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
             case  1:  {b.ReadFastArray((Char_t*)  fAddress, n); break;}
             case  2:  {b.ReadFastArray((Short_t*) fAddress, n); break;}
             case  3:  {b.ReadFastArray((Int_t*)   fAddress, n); break;}
             case  4:  {b.ReadFastArray((Long_t*)  fAddress, n); break;}
             case  5:  {b.ReadFastArray((Float_t*) fAddress, n); break;}
             case  6:  {b.ReadFastArray((Int_t*)   fAddress, n); break;}
             case  8:  {b.ReadFastArray((Double_t*)fAddress, n); break;}
             case 11:  {b.ReadFastArray((UChar_t*) fAddress, n); break;}
             case 12:  {b.ReadFastArray((UShort_t*)fAddress, n); break;}
             case 13:  {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
             case 14:  {b.ReadFastArray((ULong_t*) fAddress, n); break;}
             case 15:  {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
          }
       } else {
          fNdata = 1;
          if (fAddress) {
             fInfo->ReadBuffer(b,fObject,fID);
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
     //printf ("STL split mode not yet implemented\n");
  } else if (fType == 41) {    // sub branch of an STL class
    //char **ppointer = (char**)fAddress;
  } else if (fType == 3) {    //top level branch of a TClonesArray
    Int_t n;
    b >> n;
    fNdata = n;
    TClonesArray *clones = (TClonesArray*)fObject;
    if (!clones) return;
    clones->Clear();
    clones->ExpandCreateFast(fNdata);
  } else if (fType == 31) {    // sub branch of a TClonesArray
    fNdata = fBranchCount->GetNdata();
    TClonesArray *clones = (TClonesArray*)fObject;
    if (!clones) return;
    fInfo->ReadBufferClones(b,clones,fNdata,fID,fOffset);
  } else if (fType <= 2) {     // branch in split mode
    if (fBranchCount) fNdata = (Int_t)fBranchCount->GetValue(0,0);
    else fNdata = 1;
    fInfo->ReadBuffer(b,fObject,fID);
    if (fStreamerType == 6) fNdata = (Int_t)GetValue(0,0);
  }
}

//______________________________________________________________________________
void TBranchElement::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset a Branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//
//    Existing buffers are deleted
//    Entries, max and min are reset
//

   TBranch::Reset(option);
   
   fInfo           = gROOT->GetClass(fClassName.Data())->GetStreamerInfo(fClassVersion);
      
   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->Reset(option);
   }
}

//______________________________________________________________________________
void TBranchElement::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*Set address of this branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//

   if (TestBit(kDoNotProcess)) return;
   
   //special case when called from code generated by TTree::MakeClass
   if (Long_t(add) == -1) {
      SetBit(kWarn);
      fAddress = (char*)add;
      return;
   }
   fReadEntry = -1;

   //build the StreamerInfo if first time for the class
   TClass *cl = gROOT->GetClass(fClassName.Data());
   if (!fInfo ) GetInfo(); 
   Int_t nbranches = fBranches.GetEntriesFast();
   if (gDebug > 0) {
      printf("SetAddress, branch:%s, classname=%s, parent=%s, fID=%d, fType=%d, nbranches=%d, add=%lx, fInfo=%s, version=%d\n",GetName(),fClassName.Data(),fParentName.Data(),fID,fType,nbranches,(Long_t)add,fInfo->GetName(),fClassVersion);
   }
   fAddress = (char*)add;
   if (fTree->GetMakeClass()) {
      if (fID >= 0) {
         fObject = fAddress - fInfo->GetOffsets()[fID];
         return;
      }
   }
   if (fID < 0) {
      if (fAddress) {
         char **ppointer = (char**)fAddress;
         fObject = *ppointer;
         if (!fObject && cl) {
            //Remember if we build an object of a fake class
            if (!cl->GetClassInfo()) SetBit(kDeleteObject);
            fObject = (char*)cl->New();
            *ppointer = fObject;
         }
      } else {
         //Remember if we build an object of a fake class
         if (!cl->GetClassInfo()) SetBit(kDeleteObject);
         fObject = (char*)cl->New();
      }
      if (!fAddress) fAddress = (char*)&fObject;
   } else {
      fObject = fAddress;
   }

   //special case for a TClonesArray when address is not yet set
   //we must create the clonesarray first
   if (fType ==3) {
      if (fAddress) {
         if (fStreamerType==61) {
            // Case of an embedded ClonesArray
            fObject = fAddress;
         } else {
            TClonesArray **ppointer;
            ppointer = (TClonesArray**)fAddress;
            fObject = (char*)*ppointer;
         }
         if (!fObject) fAddress = 0;
      }
      TClass *clm = gROOT->GetClass(fClonesName.Data());
      if (clm) {
			clm->BuildRealData(); //just in case clm derives from an abstract class
			clm->GetStreamerInfo();
		}
      if (!fAddress) {
         //SetBit(kDeleteObject);
         fObject = (char*)new TClonesArray(fClonesName.Data());
         fAddress = (char*)&fObject;
      }
   }

   if (fType == 31) {
      if (fClassName != fParentName) {
         TClass *clparent = gROOT->GetClass(GetParentName());
         Int_t mOffset = 0;
         Int_t baseOffset = 0;
         TClass *clm = gROOT->GetClass(GetClassName());
         TStreamerInfo *binfo = clparent->GetStreamerInfo();
         if (clparent != clm) {
            char pname[kMaxLen];
            strcpy(pname,GetName());
            char *clast = (char*)strrchr(pname,'.');
            if (clast) {
               *clast = 0;
               char *clast2 = (char*)strrchr(pname,'.');
               if (clast2) {
                  binfo->GetStreamerElement(clast2+1,mOffset);
                  *clast2 = 0;
                  char *clast3 = (char*)strrchr(pname,'.');
                  if (clast3) {
                     TStreamerElement *el3 = binfo->GetStreamerElement(clast3+1,mOffset);
                     if (el3) {
                        Int_t mOffset2 = 0;
                        el3->GetClassPointer()->GetStreamerInfo()->GetStreamerElement(clast2+1,mOffset2);
                        mOffset += mOffset2;
                     }
                  }
               }   
               if (!clparent->GetBaseClass(clm)) fOffset = mOffset;
               if (!mOffset) {
                  if (clparent->GetBaseClass(clm)) {
                     baseOffset = clparent->GetBaseClassOffset(clm);
                     if (baseOffset < 0) baseOffset = 0; 
                     fOffset = baseOffset;                 
                  }
               }
            }
         }
      }
   }
   if (nbranches == 0) return;
   for (Int_t i=0;i<nbranches;i++)  {
      TBranch *abranch = (TBranch*)fBranches[i];
      //just in case a TBranch had been added to a TBranchElement!
      if (!abranch->InheritsFrom(TBranchElement::Class())) {
         abranch->SetAddress(fObject);
         continue;
      }
      TBranchElement *branch = (TBranchElement*)abranch;
      Int_t nb2 = branch->GetListOfBranches()->GetEntries();
      Int_t id = branch->GetID();
      Int_t mOffset = 0;
      Int_t baseOffset = 0;
      Int_t memberOffset = 0;
      TClass *clparent = gROOT->GetClass(branch->GetParentName());
      if (!clparent) clparent = cl;
      TClass *clm = gROOT->GetClass(branch->GetClassName());
      TStreamerInfo *binfo = clparent->GetStreamerInfo();
      //if sub-branch is a class deriving from the class of this branch
      //or a member of the class, one must add the base class offset
//printf("i=%d, clm=%s,clparent=%s, branch=%s, fType=%d, binfo=%s\n",i,clm->GetName(),clparent->GetName(),branch->GetName(),fType,binfo->GetName());
 
         if (clparent != clm) {
         char pname[kMaxLen];
         strcpy(pname,branch->GetName());
         char *clast = (char*)strrchr(pname,'.');
         if (clast) {
            *clast = 0;
            char *clast2 = (char*)strrchr(pname,'.');
            if (clast2) binfo->GetStreamerElement(clast2+1,mOffset);
            if (fType == 2 && !clparent->GetBaseClass(clm)) memberOffset = mOffset;
            if (fType == 1) {
               binfo->GetStreamerElement(pname,memberOffset);
            }
            if (!branch->GetType() && !mOffset) {
               if (clparent->GetBaseClass(clm)) {
                  baseOffset = clparent->GetBaseClassOffset(clm);
                  if (baseOffset < 0) baseOffset = 0;                  
               }
            }
         }
      }
      if (nb2 > 0) { 
         TStreamerInfo *info = branch->GetInfo();
         if (info) {
            Int_t *leafOffsets = info->GetOffsets();
            if (leafOffsets) {
               branch->SetAddress(fObject + leafOffsets[id] + baseOffset);
            } else {
               Error("SetAddress","info=%s, leafOffsets=0",info->GetName());
            }
         } else {
            Error("SetAddress","branch=%s, info=0",branch->GetName());
         }
      } else {
         branch->SetAddress(fObject + baseOffset +memberOffset);
      }
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
   for (Int_t i=0;i<ndata;i++) {
      elem = (TStreamerElement*)elems[i];
     if (gDebug > 1) printf("Unroll name=%s, cltop=%s, cl=%s, i=%d, elem=%s, splitlevel=%d, btype=%d \n",name,cltop->GetName(),cl->GetName(),i,elem->GetName(),splitlevel,btype);
     if (elem->IsA() == TStreamerBase::Class()) {
         clbase = gROOT->GetClass(elem->GetName());
         //here one should consider the case of a TClonesArray with a class
         //deriving from an abstract class
         //if ((cltop != cl) && (clbase->Property() & kIsAbstract)) return -1;
         //if (clbase->Property() & kIsAbstract) return -1;
         if (clbase->Property() & kIsAbstract) {
            if (cl->InheritsFrom("TCollection")) unroll = -1;
         }
         if (gDebug > 1) printf("Unrolling base class, cltop=%s, clbase=%s\n",cltop->GetName(),clbase->GetName());
         if (unroll < 0 && btype != 31) return -1;
         else unroll = Unroll(name,cltop,clbase,basketsize,splitlevel-1,btype);
         if (unroll < 0) {
            if (strlen(name)) sprintf(branchname,"%s.%s",name,elem->GetFullName());
            else              sprintf(branchname,"%s",elem->GetFullName());
            branch = new TBranchElement(branchname,info,jd,0,basketsize,0,btype);
            branch->SetParentName(cltop->GetName());
            fBranches.Add(branch);
         }
      } else {
        if (strlen(name)) sprintf(branchname,"%s.%s",name,elem->GetFullName());
        else              sprintf(branchname,"%s",elem->GetFullName());
        if (splitlevel > 1 &&
              (elem->IsA() == TStreamerObject::Class()
            || elem->IsA() == TStreamerObjectAny::Class())) {
               clbase = gROOT->GetClass(elem->GetTypeName());
               if (clbase->Property() & kIsAbstract) return -1;

            if (gDebug > 1) printf("Unrolling object class, cltop=%s, clbase=%s\n",cltop->GetName(),clbase->GetName());
            if (elem->CannotSplit())    unroll = -1;
            else unroll = Unroll(branchname,cltop,clbase,basketsize,splitlevel-1,btype);
            if (unroll < 0) {
               branch = new TBranchElement(branchname,info,jd,0,basketsize,0,btype);
               branch->SetParentName(cltop->GetName());
               fBranches.Add(branch);
            }
         } else {
            if (elem->GetClassPointer() == TClonesArray::Class()) {
               //process case of a TClonesArray in a derived class (do not split)
               char *pointer = fBranchPointer + elem->GetOffset();
               //branch = new TBranchElement(branchname,info,jd,pointer,basketsize,splitlevel-1,btype);
               branch = new TBranchElement(branchname,info,jd,pointer,basketsize,0,btype);
            } else {
               branch = new TBranchElement(branchname,info,jd,0,basketsize,0,btype);
               branch->SetType(btype);
            }
            branch->SetParentName(cltop->GetName());
            fBranches.Add(branch);
         }
      }
      jd++;
   }
   return 1;
}
