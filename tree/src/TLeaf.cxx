// @(#)root/tree:$Name:  $:$Id: TLeaf.cxx,v 1.6 2001/11/17 15:56:00 brun Exp $
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TLeaf describes individual elements of a TBranch                   //
//       See TBranch structure in TTree.                                //
//////////////////////////////////////////////////////////////////////////

#include "TLeaf.h"
#include "TBranch.h"
#include "TTree.h"
#include "TVirtualPad.h"
#include "TBrowser.h"

#include <ctype.h>

R__EXTERN TTree *gTree;
R__EXTERN TBranch *gBranch;


ClassImp(TLeaf)

//______________________________________________________________________________
TLeaf::TLeaf(): TNamed()
{
//*-*-*-*-*-*Default constructor for Leaf*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ============================
   fLen        = 0;
   //fBranch     = 0;
   fBranch     = gBranch;
   fLeafCount  = 0;
   fNdata      = 0;
   fOffset     = 0;
}

//______________________________________________________________________________
TLeaf::TLeaf(const char *name, const char *)
    :TNamed(name,name)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a Leaf*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============
//
//     See the TTree and TBranch constructors for explanation of parameters.

   fLeafCount  = GetLeafCounter(fLen);
   if (fLen == -1) {MakeZombie(); return;}
   fIsRange    = 0;
   fIsUnsigned = 0;
   fLenType    = 4;
   fNdata      = 0;
   fOffset     = 0;
   if (fLeafCount || strchr(name,'[')) {
      char newname[64];
      strcpy(newname,name);
      char *bracket = strchr(newname,'[');
      *bracket = 0;
      SetName(newname);
   }
   fBranch     = gBranch;
}

//______________________________________________________________________________
TLeaf::~TLeaf()
{
//*-*-*-*-*-*Default destructor for a Leaf*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ===============================

//   if (fBranch) fBranch->GetListOfLeaves().Remove(this);
   if (!fBranch) return;
   TTree *tree = fBranch->GetTree();
   fBranch = 0;
   if (!tree) return;
   tree->GetListOfLeaves()->Remove(this);
}



//______________________________________________________________________________
void TLeaf::Browse(TBrowser *)
{
   char name[64];
   if (strchr(GetName(),'.')) {
      fBranch->GetTree()->Draw(GetName());
   } else {
      sprintf(name,"%s.%s",fBranch->GetName(),GetName());
      fBranch->GetTree()->Draw(name);
   }
   if (gPad) gPad->Update();
}


//______________________________________________________________________________
void TLeaf::FillBasket(TBuffer &)
{
//*-*-*-*-*-*-*-*-*-*-*Pack leaf elements in Basket output buffer*-*-*-*-*-*-*
//*-*                  =========================================

}

//______________________________________________________________________________
TLeaf *TLeaf::GetLeafCounter(Int_t &countval) const
{
//*-*-*-*-*-*-*Return Pointer to counter of referenced Leaf*-*-*-*-*-*-*-*
//*-*          ============================================
//
//  If leaf name has the forme var[nelem], where nelem is alphanumeric, then
//     If nelem is a leaf name, return countval = 1 and the pointer to 
//     the leaf named nelem.
//  If leaf name has the forme var[nelem], where nelem is a digit, then
//     return countval = nelem and a null pointer.
//  Otherwise return countval=0 and a null pointer.
//

   countval = 1;
   const char *name = GetTitle();
   char *bleft = (char*)strchr(name,'[');
   if (!bleft) return 0;
   bleft++;
   Int_t nch = strlen(bleft);
   char *countname = new char[nch+1];
   strcpy(countname,bleft);
   char *bright = (char*)strchr(countname,']');
   if (!bright) { delete [] countname; return 0;}
   char *bleft2 = (char*)strchr(countname,'[');
   *bright = 0; nch = strlen(countname);

//*-* Now search a branch name with a leave name = countname
  TLeaf *leaf = (TLeaf*)gTree->GetListOfLeaves()->FindObject(countname);
  Int_t i;
  if (leaf) {
     countval = 1;
     leaf->SetRange();
     if (bleft2) {
        sscanf(bleft2,"[%d]",&i);
        countval *= i;
     }
     bleft = bleft2;
     while(bleft) {
        bleft2++;
        bleft = (char*)strchr(bleft2,'[');
        if (!bleft) break;
        sscanf(bleft,"[%d]",&i);
        countval *= i;
        bleft2 = bleft;
     }
     delete [] countname;
     return leaf;
  }
//*-* not found in a branch/leaf. Is it a numerical value?
   for (i=0;i<nch;i++) {
      if (!isdigit(countname[i])) {
        delete [] countname;
        countval = -1;
        return 0;
      }
   }
   sscanf(countname,"%d",&countval);
   if (bleft2) {
      sscanf(bleft2,"[%d]",&i);
      countval *= i;
   }
   bleft = bleft2;
   while(bleft) {
      bleft2++;
      bleft = (char*)strchr(bleft2,'[');
      if (!bleft) break;
      sscanf(bleft,"[%d]",&i);
      countval *= i;
      bleft2 = bleft;
   }
//*/
   delete [] countname;
   return 0;
}


//______________________________________________________________________________
Int_t TLeaf::GetLen() const
{
//*-*-*-*-*-*-*-*-*Return the number of effective elements of this leaf*-*-*-*
//*-*              ====================================================

   Int_t len;
   if (fLeafCount) {
      len = Int_t(fLeafCount->GetValue());
      if (len > fLeafCount->GetMaximum()) {
         printf("ERROR leaf:%s, len=%d and max=%d\n",GetName(),len,fLeafCount->GetMaximum());
         len = fLeafCount->GetMaximum();
      }
      return len*fLen;
   } else {
      return fLen;
   }
}

//______________________________________________________________________________
Int_t TLeaf::ResetAddress(void *add, Bool_t destructor)
{
//*-*-*-*-*-*-*-*-*-*-*Set leaf buffer data address*-*-*-*-*-*
//*-*                  ============================
//
//  This function is called by all TLeafX::SetAddress


   Int_t todelete = 0;
   if (TestBit(kNewValue)) todelete = 1;
   if (destructor) return todelete;

   if (fLeafCount) fNdata = fLen*(fLeafCount->GetMaximum() + 1);
   else            fNdata = fLen;

   ResetBit(kNewValue);
   if (!add) SetBit(kNewValue);
   return todelete;
}

//_______________________________________________________________________
void TLeaf::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================

   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TLeaf::Class()->ReadBuffer(b, this, R__v, R__s, R__c);
      } else {
         //====process old versions before automatic schema evolution
         TNamed::Streamer(b);
         b >> fLen;
         b >> fLenType;
         b >> fOffset;
         b >> fIsRange;
         b >> fIsUnsigned;
         b >> fLeafCount;
         b.CheckByteCount(R__s, R__c, TLeaf::IsA());
         //====end of old versions
      }
      if (fLen == 0) fLen = 1;
      ResetBit(kNewValue);
      SetAddress();
   } else {
      TLeaf::Class()->WriteBuffer(b,this);
   }
}
