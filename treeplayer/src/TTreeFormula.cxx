// @(#)root/treeplayer:$Name:  $:$Id: TTreeFormula.cxx,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Rene Brun   19/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TTreeFormula.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeafC.h"
#include "TLeafObject.h"
#include "TMethodCall.h"
#include "TCutG.h"

#include <stdio.h>
#include <math.h>

const Int_t kMETHOD = 1000;

ClassImp(TTreeFormula)

//______________________________________________________________________________
//
//     A TreeFormula is used to pass a selection expression
//     to the Tree drawing routine. See TTree::Draw
//
//  A TreeFormula can contain any arithmetic expression including
//  standard operators and mathematical functions separated by operators.
//  Examples of valid expression:
//          "x<y && sqrt(z)>3.2"
//


//______________________________________________________________________________
TTreeFormula::TTreeFormula(): TFormula()
{
//*-*-*-*-*-*-*-*-*-*-*Tree Formula default constructor*-*-*-*-*-*-*-*-*-*
//*-*                  ================================

   fTree   = 0;
   fIndex  = 0;
   fNindex = 0;
   fNcodes = 0;
}

//______________________________________________________________________________
TTreeFormula::TTreeFormula(const char *name,const char *expression, TTree *tree)
               :TFormula()
{
//*-*-*-*-*-*-*-*-*-*-*Normal Tree Formula constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
//

   fTree         = tree;
   fNindex       = 100;
   fIndex        = new Int_t[fNindex];
   fNcodes       = 0;
   fMultiplicity = 0;

   if (Compile(expression)) {fTree = 0; return; }
   SetName(name);
   for (Int_t i=0;i<fNcodes;i++) {
      if (fCodes[i] < 0) continue;
      TLeaf *leaf = GetLeaf(i);
      if (leaf->InheritsFrom("TLeafC")) SetBit(kIsCharacter);
      if (leaf->InheritsFrom("TLeafB")) SetBit(kIsCharacter);
      if(leaf->GetLeafCount() && fIndex[i] < 0) fMultiplicity = 1;
      if(leaf->GetLeafCount() && fMultiplicity == 0) fMultiplicity = -1;
      if (fIndex[i] == -1 ) fIndex[i] = 0;
   }
}

//______________________________________________________________________________
TTreeFormula::~TTreeFormula()
{
//*-*-*-*-*-*-*-*-*-*-*Tree Formula default destructor*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================================

   if (fIndex) delete [] fIndex;
}

//______________________________________________________________________________
Int_t TTreeFormula::DefinedVariable(TString &name)
{
//*-*-*-*-*-*Check if name is in the list of Tree/Branch leaves*-*-*-*-*
//*-*        ==================================================
//
//   This member function redefines the function in TFormula
//   If a leaf has a name corresponding to the argument name, then
//   returns a new code.
//   A TTreeFormula may contain more than one variable.
//   For each variable referenced, the pointers to the corresponding
//   branch and leaf is stored in the object arrays fBranches and fLeaves.
//
//   name can be :
//      - Leaf_Name (simple variable or data member of a ClonesArray)
//      - Branch_Name.Leaf_Name
//      - Branch_Name.Method_Name
//

   if (!fTree) return -1;
   fNpar = 0;
   Int_t nchname = name.Length();
   if (nchname > 60) return -1;
   char branchname[128];
   static char anumber[10];
   static char lname[64];
   static char lmethod[128];
   Int_t i;
   strcpy(lname,name.Data());
   Int_t dot   = 0;
   Int_t index = -1;
   Int_t scanindex = 0;
   char *ref = 0;
   for (i=0;i<nchname;i++) {
      if (lname[i] == '.') {
         if (i == nchname-1) break; // for example 999.
         if (!dot) {
            dot = i;
            strcpy(lmethod,&lname[i+1]);
         }
      }
      if (name[i] == '[' && name[nchname-1] == ']') {
         strncpy(anumber,&name[i+1],nchname-i-2);
         anumber[nchname-i-2] = 0;
         scanindex = sscanf(anumber,"%d",&index);
         lname[i] = 0;
         if (scanindex) {lname[i] = 0; break;}
         if (ref == 0) ref= anumber;
         index = 0;
      }
   }
   TObjArray *lleaves = fTree->GetListOfLeaves();
   Int_t nleaves = lleaves->GetEntriesFast();

//*-*- There is a dot. two cases: data member or member function
   if (dot) {
//          Look for a data member
      for (i=0;i<nleaves;i++) {
         TLeaf *leaf = (TLeaf*)lleaves->UncheckedAt(i);
         if (!strcmp(lname,leaf->GetBranch()->GetName() ) ) {
            TMethodCall *method = 0;
            fMethods.Add(method);
            Int_t code = fNcodes;
            fCodes[code] = i;
            fNcodes++;
            if (scanindex) fIndex[code] = index;
            else           fIndex[code] = -1;
            if (leaf->InheritsFrom("TLeafC")) return 5000+code;
            return code;
         }
      }
//          Look for branchname.leafname
      for (i=0;i<nleaves;i++) {
         TLeaf *leaf = (TLeaf*)lleaves->UncheckedAt(i);
         sprintf(branchname,"%s.%s",leaf->GetBranch()->GetName(),leaf->GetName());
         if (!strcmp(lname,branchname)) {
            TMethodCall *method = 0;
            fMethods.Add(method);
            Int_t code = fNcodes;
            fCodes[code] = i;
            fNcodes++;
            if (scanindex) fIndex[code] = index;
            else           fIndex[code] = -1;
            if (leaf->InheritsFrom("TLeafC")) return 5000+code;
            return code;
         }
      }
//          Look for a member function
      for (i=0;i<nleaves;i++) {
         lname[dot] = 0;
         TLeaf *leaf = (TLeaf*)lleaves->UncheckedAt(i);
         if (!strcmp(lname,leaf->GetBranch()->GetName())) {
            if (leaf->IsA() != TLeafObject::Class()) return -1;
            TLeafObject *lobj = (TLeafObject*)leaf;
            TMethodCall *method = lobj->GetMethodCall(lmethod);
            if (!method) return -1;
            fMethods.Add(method);
            Int_t code = fNcodes;
            fCodes[code] = i;
            fNcodes++;
            fIndex[code] = code-kMETHOD;
            return code;
         }
      }

//*-*- There is no dot. Look for a data member only
   } else {
      for (i=0;i<nleaves;i++) {
         TLeaf *leaf = (TLeaf*)lleaves->UncheckedAt(i);
         if (!strcmp(lname,leaf->GetBranch()->GetName()) ||
             !strcmp(lname,leaf->GetName())) {
            TMethodCall *method = 0;
            fMethods.Add(method);
            Int_t code = fNcodes;
            fCodes[code] = i;
            fNcodes++;
            if (scanindex) fIndex[code] = index;
            else           fIndex[code] = -1;
            if (ref) printf("Cannot process reference to array index=%s[%s]\n",lname,ref);
            if (leaf->InheritsFrom("TLeafC"))                   return 5000+code;
            if (leaf->InheritsFrom("TLeafB") && scanindex == 0) return 5000+code;
            return code;
         }
      }
   }

//*-*- May be a graphical cut ?
   TCutG *gcut = (TCutG*)gROOT->GetListOfSpecials()->FindObject(name.Data());
   if (gcut) {
      if (!gcut->GetObjectX()) {
         TTreeFormula *fx = new TTreeFormula("f_x",gcut->GetVarX(),fTree);
         gcut->SetObjectX(fx);
      }
      if (!gcut->GetObjectY()) {
         TTreeFormula *fy = new TTreeFormula("f_y",gcut->GetVarY(),fTree);
         gcut->SetObjectY(fy);
      }
      //these 3 lines added by Romain.Holzmann@gsi.de
      Int_t mulx = ((TTreeFormula*)gcut->GetObjectX())->GetMultiplicity();
      Int_t muly = ((TTreeFormula*)gcut->GetObjectY())->GetMultiplicity();
      if(mulx || muly) fMultiplicity = -1;

      fMethods.Add(gcut);
      Int_t code = fNcodes;
      fCodes[code] = -1;
      fNcodes++;
      fIndex[code] = -1;
      return code;
   }
   return -1;
}

//______________________________________________________________________________
Double_t TTreeFormula::EvalInstance(Int_t instance)
{
//*-*-*-*-*-*-*-*-*-*-*Evaluate this treeformula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//

  Int_t i,pos,pos2,int1,int2;
  Float_t aresult;
  Double_t tab[255];
  Float_t param[50];
  Double_t dexp;
  char *tab2[20];

  if (fNoper == 1 && fNcodes > 0) {
     if (fCodes[0] < 0) {
        TCutG *gcut = (TCutG*)fMethods.At(0);
        TTreeFormula *fx = (TTreeFormula *)gcut->GetObjectX();
        TTreeFormula *fy = (TTreeFormula *)gcut->GetObjectY();
        Float_t xcut = fx->EvalInstance(instance);
        Float_t ycut = fy->EvalInstance(instance);
        return gcut->IsInside(xcut,ycut);
     }
     TLeaf *leaf = GetLeaf(0);
     if (instance) {
        if (instance < leaf->GetNdata()) return leaf->GetValue(instance);
        else                             return leaf->GetValue(0);
     } else {
        leaf->GetBranch()->GetEntry(fTree->GetReadEntry());
        if (!(leaf->IsA() == TLeafObject::Class())) return leaf->GetValue(fIndex[0]);
        return GetValueLeafObject(fIndex[0],(TLeafObject *)leaf);
     }
  }
  for(i=0;i<fNval;i++) {
     if (fCodes[i] < 0) {
        TCutG *gcut = (TCutG*)fMethods.At(i);
        TTreeFormula *fx = (TTreeFormula *)gcut->GetObjectX();
        TTreeFormula *fy = (TTreeFormula *)gcut->GetObjectY();
        Float_t xcut = fx->EvalInstance(instance);
        Float_t ycut = fy->EvalInstance(instance);
        param[i] = gcut->IsInside(xcut,ycut);
     } else {
        TLeaf *leaf = GetLeaf(i);
        if (instance) {
           if (instance < leaf->GetNdata()) param[i] = leaf->GetValue(instance);
           else                             param[i] = leaf->GetValue(0);
        } else {
           leaf->GetBranch()->GetEntry(fTree->GetReadEntry());
           if (!(leaf->IsA() == TLeafObject::Class())) param[i] = leaf->GetValue(fIndex[i]);
           else param[i] = GetValueLeafObject(fIndex[i],(TLeafObject *)leaf);
        }
     }
  }
  pos  = 0;
  pos2 = 0;
  for (i=0; i<fNoper; i++) {
    Int_t action = fOper[i];
//*-*- a tree string
    if (action >= 105000) {
       TLeaf *leafc = GetLeaf(action-105000);
       leafc->GetBranch()->GetEntry(fTree->GetReadEntry());
       pos2++; tab2[pos2-1] = (char*)leafc->GetValuePointer();
       continue;
    }
//*-*- a tree variable
    if (action >= 100000) {
       pos++; tab[pos-1] = param[action-100000];
       continue;
    }
//*-*- String
    if (action == 80000) {
       pos2++; tab2[pos2-1] = (char*)fExpr[i].Data();
       continue;
    }
//*-*- numerical value
    if (action >= 50000) {
       pos++; tab[pos-1] = fConst[action-50000];
       continue;
    }
    if (action == 0) {
      pos++;
      sscanf((const char*)fExpr[i],"%g",&aresult);
      tab[pos-1] = aresult;
//*-*- basic operators and mathematical library
    } else if (action < 100) {
        switch(action) {
          case   1 : pos--; tab[pos-1] += tab[pos]; break;
          case   2 : pos--; tab[pos-1] -= tab[pos]; break;
          case   3 : pos--; tab[pos-1] *= tab[pos]; break;
          case   4 : pos--; if (tab[pos] == 0) tab[pos-1] = 0; //  division by 0
                            else               tab[pos-1] /= tab[pos];
                     break;
          case   5 : {pos--; int1=Int_t(tab[pos-1]); int2=Int_t(tab[pos]); tab[pos-1] = Double_t(int1%int2); break;}
          case  10 : tab[pos-1] = TMath::Cos(tab[pos-1]); break;
          case  11 : tab[pos-1] = TMath::Sin(tab[pos-1]); break;
          case  12 : if (TMath::Cos(tab[pos-1]) == 0) {tab[pos-1] = 0;} // { tangente indeterminee }
                     else tab[pos-1] = TMath::Tan(tab[pos-1]);
                     break;
          case  13 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} //  indetermination
                     else tab[pos-1] = TMath::ACos(tab[pos-1]);
                     break;
          case  14 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} //  indetermination
                     else tab[pos-1] = TMath::ASin(tab[pos-1]);
                     break;
          case  15 : tab[pos-1] = TMath::ATan(tab[pos-1]); break;
          case  70 : tab[pos-1] = TMath::CosH(tab[pos-1]); break;
          case  71 : tab[pos-1] = TMath::SinH(tab[pos-1]); break;
          case  72 : if (TMath::CosH(tab[pos-1]) == 0) {tab[pos-1] = 0;} // { tangente indeterminee }
                     else tab[pos-1] = TMath::TanH(tab[pos-1]);
                     break;
          case  73 : if (tab[pos-1] < 1) {tab[pos-1] = 0;} //  indetermination
                     else tab[pos-1] = TMath::ACosH(tab[pos-1]);
                     break;
          case  74 : tab[pos-1] = TMath::ASinH(tab[pos-1]); break;
          case  75 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} // indetermination
                     else tab[pos-1] = TMath::ATanH(tab[pos-1]); break;
          case  16 : pos--; tab[pos-1] = TMath::ATan2(tab[pos-1],tab[pos]); break;
          case  17 : pos--; tab[pos-1] = fmod(tab[pos-1],tab[pos]); break;
          case  20 : pos--; tab[pos-1] = TMath::Power(tab[pos-1],tab[pos]); break;
          case  21 : tab[pos-1] = tab[pos-1]*tab[pos-1]; break;
          case  22 : tab[pos-1] = TMath::Sqrt(TMath::Abs(tab[pos-1])); break;
          case  23 : pos2 -= 2; pos++;if (strstr(tab2[pos2],tab2[pos2+1])) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  30 : if (tab[pos-1] > 0) tab[pos-1] = TMath::Log(tab[pos-1]);
                     else {tab[pos-1] = 0;} //{indetermination }
                     break;
          case  31 : dexp = tab[pos-1];
                     if (dexp < -70) {tab[pos-1] = 0; break;}
                     if (dexp >  70) {tab[pos-1] = TMath::Exp(70); break;}
                     tab[pos-1] = TMath::Exp(dexp); break;
          case  32 : if (tab[pos-1] > 0) tab[pos-1] = TMath::Log10(tab[pos-1]);
                     else {tab[pos-1] = 0;} //{indetermination }
                     break;
          case  40 : pos++; tab[pos-1] = TMath::ACos(-1); break;
          case  41 : tab[pos-1] = TMath::Abs(tab[pos-1]); break;
          case  42 : if (tab[pos-1] < 0) tab[pos-1] = -1; else tab[pos-1] = 1; break;
          case  43 : tab[pos-1] = Double_t(Int_t(tab[pos-1])); break;
      //  case  50 : tab[pos-1] = gRandom->Rndm(1); break;
          case  50 : tab[pos-1] = 0.5; break;
          case  60 : pos--; if (tab[pos-1]!=0 && tab[pos]!=0) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  61 : pos--; if (tab[pos-1]!=0 || tab[pos]!=0) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  62 : pos--; if (tab[pos-1] == tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  63 : pos--; if (tab[pos-1] != tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  64 : pos--; if (tab[pos-1] < tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  65 : pos--; if (tab[pos-1] > tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  66 : pos--; if (tab[pos-1]<=tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  67 : pos--; if (tab[pos-1]>=tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  68 : if (tab[pos-1]!=0) tab[pos-1] = 0; else tab[pos-1] = 1; break;
          case  76 : pos2 -= 2; pos++; if (!strcmp(tab2[pos2+1],tab2[pos2])) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  77 : pos2 -= 2; pos++;if (strcmp(tab2[pos2+1],tab2[pos2])) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  78 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) & ((Int_t) tab[pos]); break;
          case  79 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) | ((Int_t) tab[pos]); break;
       }
    }
  }
  Double_t result = tab[0];
  return result;
}

//______________________________________________________________________________
TLeaf *TTreeFormula::GetLeaf(Int_t n)
{
//*-*-*-*-*-*-*-*Return leaf corresponding to serial number n*-*-*-*-*-*
//*-*            ============================================
//

   return (TLeaf*)fTree->GetListOfLeaves()->UncheckedAt(fCodes[n]);
}

//______________________________________________________________________________
TMethodCall *TTreeFormula::GetMethodCall(Int_t code)
{
//*-*-*-*-*-*-*-*Return methodcall corresponding to code*-*-*-*-*-*
//*-*            =======================================
//
//  function called by TLeafObject::GetValue
//  with the value of fIndex computed in TTreeFormula::DefinedVariable

   return (TMethodCall *)fMethods.UncheckedAt(code+kMETHOD);

}

//______________________________________________________________________________
Int_t TTreeFormula::GetNdata()
{
//*-*-*-*-*-*-*-*Return number of data words in the leaf*-*-*-*-*-*-*-*
//*-*            =======================================
//

   for (Int_t i=0;i<fNcodes;i++) {
      if (fCodes[i] < 0) continue;
      TLeaf *leaf = GetLeaf(i);
      if (leaf->GetLeafCount()) {
         TBranch *branch = leaf->GetLeafCount()->GetBranch();
         branch->GetEntry(fTree->GetReadEntry());
         return leaf->GetLen();
      }
   }
   return 0;
}

//______________________________________________________________________________
Float_t TTreeFormula::GetValueLeafObject(Int_t i, TLeafObject *leaf)
{
//*-*-*-*-*-*-*-*Return result of a leafobject method*-*-*-*-*-*-*-*
//*-*            ====================================
//
   TMethodCall *m = GetMethodCall(i);
   if (!m) return 0;

   void *thisobj = leaf->GetObject();

   EReturnType r = m->ReturnType();

   if (r == kLong) {
      Long_t l;
      m->Execute(thisobj, l);
      return (Float_t) l;
   }
   if (r == kDouble) {
      Double_t d;
      m->Execute(thisobj, d);
      return (Float_t) d;
   }
   m->Execute(thisobj);
   return 0;
}

//______________________________________________________________________________
char *TTreeFormula::PrintValue(Int_t mode)
{
//*-*-*-*-*-*-*-*Return value of variable as a string*-*-*-*-*-*-*-*
//*-*            ====================================
//
//      mode = -2 : Print line with ***
//      mode = -1 : Print column names
//      mode = 0  : Print column values

   const int kMAXLENGTH = 1024;
   static char value[kMAXLENGTH];

   if (mode == -2) {
      for (int i = 0; i < kMAXLENGTH-1; i++)
         value[i] = '*';
      value[kMAXLENGTH-1] = 0;
   } else if (mode == -1)
      sprintf(value, "%s", GetTitle());

   if (TestBit(kIsCharacter)) {
      if (mode == 0) {
         TLeaf *leaf = GetLeaf(0);
         leaf->GetBranch()->GetEntry(fTree->GetReadEntry());
         strncpy(value, (char*)leaf->GetValuePointer(), kMAXLENGTH-1);
         value[kMAXLENGTH-1] = 0;
      }
   } else {
      if (mode == 0) {
         GetNdata();
         sprintf(value,"%9.5g",EvalInstance(0));
      }
   }
   return &value[0];
}

//______________________________________________________________________________
void TTreeFormula::Streamer(TBuffer &R__b)
{
   // Stream an object of class TTreeFormula.

   if (R__b.IsReading()) {
      R__b.ReadVersion();  //Version_t R__v = R__b.ReadVersion();
      TFormula::Streamer(R__b);
      R__b >> fTree;
      R__b >> fNcodes;
      R__b.ReadFastArray(fCodes, fNcodes);
      R__b >> fMultiplicity;
      R__b >> fInstance;
      R__b >> fNindex;
      if (fNindex) {
         fIndex = new Int_t[fNindex];
         R__b.ReadFastArray(fIndex, fNindex);
      }
      fMethods.Streamer(R__b);
   } else {
      R__b.WriteVersion(TTreeFormula::IsA());
      TFormula::Streamer(R__b);
      R__b << fTree;
      R__b << fNcodes;
      R__b.WriteFastArray(fCodes, fNcodes);
      R__b << fMultiplicity;
      R__b << fInstance;
      R__b << fNindex;
      if (fNindex) R__b.WriteFastArray(fIndex, fNindex);
      fMethods.Streamer(R__b);
   }
}
