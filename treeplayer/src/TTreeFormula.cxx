// @(#)root/treeplayer:$Name:  $:$Id: TTreeFormula.cxx,v 1.19 2000/11/21 20:58:28 brun Exp $
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
#include "TRandom.h"

#include <stdio.h>
#include <math.h>

const Int_t kMETHOD   = 1000;

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
   fNindex       = kMAXFOUND;
   fIndex        = new Int_t[fNindex];
   fNcodes       = 0;
   fMultiplicity = 0;
   Int_t i,j,k;

   for (j=0; j<kMAXCODES; j++) {
      fNdimensions[j] = 0;
      for (k = 0; k<kMAXFORMDIM; k++) {
         fIndexes[j][k] = -1;
         fCumulSize[j][k] = 1;
      }
   }
   for (k = 0; k<kMAXFORMDIM+1; k++) {
      fCumulUsedSize[k] = 1;
   }

   if (Compile(expression)) {fTree = 0; return; }
   if (fNcodes >= kMAXFOUND) {
      Warning("TTreeFormula","Too many items in expression:%s",expression);
      fNcodes = kMAXFOUND;
   }
   SetName(name);
   for (i=0;i<fNcodes;i++) {
      if (fCodes[i] < 0) continue;
      TLeaf *leaf = GetLeaf(i);
      if (leaf->InheritsFrom("TLeafC") && !leaf->IsUnsigned()) SetBit(kIsCharacter);
      if (leaf->InheritsFrom("TLeafB") && !leaf->IsUnsigned()) SetBit(kIsCharacter);

      // Reminder of the meaning of fMultiplicity:
      //  -1: Only one or 0 element per entry, contains variable length array!
      //   0: Only one element per entry, no variable length array
      //   1: loop over the elements of a variable length array
      //   2: loop over elements of fixed length array (nData is the same for all entry)

      if (leaf->GetLeafCount()) {
         // We assume only one possible variable length dimension (the left most)
         fMultiplicity = 1;
      } else {
         if (leaf->GetLenStatic()>1 && fMultiplicity!=1) fMultiplicity = 2;
      }

      if (fIndex[i] == -1 ) fIndex[i] = 0;

      // Because we did not record the number of virtual dimensions associated
      // with this leaf, we can not use the next loop which has to count down.
      Int_t virt_dim = 0;
      for (k = 0; k < fNdimensions[i]; k++) {
         if ( (fCumulSize[i][k]>=0) && (fIndexes[i][k] >= fCumulSize[i][k]) ) {
            // unreacheable element requested:
            fCumulUsedSize[virt_dim] = 0;
         }
         if ( fIndexes[i][k] < 0 ) virt_dim++;
      }
      // Add up the cumulative size
      for (k = fNdimensions[i]; (k > 0) && (fCumulSize[i][k-1]>=0); k--) {
         fCumulSize[i][k-1] *= fCumulSize[i][k];
      }

   }
   for (k = kMAXFORMDIM; (k > 0) && (fCumulUsedSize[k-1]>=0); k--) {
      fCumulUsedSize[k-1] *= fCumulUsedSize[k];
   }
   // Now that we know the virtual dimension we know if a loop over EvalInstance
   // is needed or not.
   if (fCumulUsedSize[0]==1 && fMultiplicity!=0) fMultiplicity -= 2;

}

//______________________________________________________________________________
TTreeFormula::~TTreeFormula()
{
//*-*-*-*-*-*-*-*-*-*-*Tree Formula default destructor*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================================

   if (fIndex) delete [] fIndex;
}

//______________________________________________________________________________
void TTreeFormula::DefineDimensions(const char *info, Int_t code, Int_t& virt_dim) {
   // We assume that there are NO white spaces in the info string
   const char * current;
   Int_t size, scanindex;
   Bool_t neg;

   current = info;
   // the next value could be before the string but
   // that's okay because the next operation is ++
   // (this is to avoid (?) a if statement at the end of the
   // loop
   if (current[0] != '[') current--;
   while (current) {
      current++;
      scanindex = sscanf(current,"%d",&size);
      // if scanindex is 0 then we have a name index thus a variable
      // array (or TClonesArray!).
      if (scanindex==0) {
         fCumulSize[code][fNdimensions[code]] = -1;
         if ( fIndexes[code][fNdimensions[code]] < 0 ) {
            fCumulUsedSize[virt_dim] = -1 * TMath::Abs(fCumulUsedSize[virt_dim]);
            virt_dim++;
         }
      } else {
         fCumulSize[code][fNdimensions[code]] = size;
         if ( fIndexes[code][fNdimensions[code]] < 0 ) {
            if ( TMath::Abs(fCumulUsedSize[virt_dim])==1
                 || (size < TMath::Abs(fCumulUsedSize[virt_dim]) ) ) {
               neg = fCumulUsedSize[virt_dim] < 0;
               fCumulUsedSize[virt_dim] = size;
               if (neg) fCumulUsedSize[virt_dim] *= -1;
            }
            virt_dim++;
         }
      }
      fNdimensions[code] ++;
      if (fNdimensions[code] >= kMAXFORMDIM) {
         // NOTE: test that fNdimensions[code] this is NOT too big!!

         break;
      }
      current = (char*)strstr( current, "[" );
   }

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
//      - Leaf_Name[index]
//      - Branch_Name.Leaf_Name[index]
// I want to support, with Leaf_Name a 1D array data member.
//      - Branch_Name.Leaf_Name[index1]
//      - Branch_Name.Leaf_Name[][index2]
//      - Branch_Name.Leaf_Name[index1][index2]
//

   if (!fTree) return -1;
   fNpar = 0;
   Int_t nchname = name.Length();
   if (nchname > 60) return -1;
   char branchname[128];
   char leafname[128];
   static char anumber[10];
   static char lname[128];
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
         char * current = &( name[i] );
         Int_t dim = 0;
         while (current) {
            current++;
            if (current[0] == ']') {
               fIndexes[fNcodes][dim] = -1; // Loop over all elements;
            } else {
               scanindex = sscanf(current,"%d",&index);
               if (scanindex) {
                  fIndexes[fNcodes][dim] = index;
               } else {
                  if (ref == 0) ref= anumber;
                  index = 0;
               }
            }
            dim ++;
            if (dim >= kMAXFORMDIM) {
               // NOTE: test that dim this is NOT too big!!
               break;
            }
            current = (char*)strstr( current, "[" );
         }
         lname[i] = 0;
         break;
      }
   }
   TObjArray *lleaves = fTree->GetListOfLeaves();
   Int_t nleaves = lleaves->GetEntriesFast();

//*-*- There is a dot. two cases: data member or member function
   if (dot) {
//          Look for a data member
      for (i=0;i<nleaves;i++) {
         TLeaf *leaf = (TLeaf*)lleaves->UncheckedAt(i);
         if (!leaf->GetBranch()) continue;
         strcpy(branchname,leaf->GetBranch()->GetName());
         // do not look at the indexes if any
         char *branch_dim = (char*)strstr(branchname,"[");
         if (branch_dim) { branch_dim[0] = '\0'; branch_dim++; }
         if (!strcmp(lname, branchname) ) {
            TMethodCall *method = 0;
            fMethods.Add(method);
            Int_t code = fNcodes;
            fCodes[code] = i;
            fNcodes++;
            if (scanindex) fIndex[code] = index;
            else           fIndex[code] = -1;

            // Let see if we can understand the structure of this branch.
            // Usually we have: leafname[fixed_array] leaftitle[var_array]\type
            // (with fixed_array that can be a multi-dimension array.
            // NOTE I removed this? if ( !strcmp(lname,leaf->GetName() ) ) {
            const char *tname = leaf->GetTitle();
            char *dim = (char*)strstr( tname, "[" );
            Int_t virt_dim = 0;
            if (dim) {
               dim++;
               if (!branch_dim || strncmp(branch_dim,dim,strlen(branch_dim))) {
                  // then both are NOT the same so do the title first:
                  DefineDimensions( dim, code, virt_dim);
               }
            }
            if (branch_dim) {
               // then both are NOT same so do the title:
               DefineDimensions( branch_dim, code, virt_dim);
            }
            // should we also check the leaf name?

            if (leaf->InheritsFrom("TLeafC") && !leaf->IsUnsigned()) return 5000+code;
            return code;
         }
      }
//          Look for branchname.leafname
      for (i=0;i<nleaves;i++) {
         TLeaf *leaf = (TLeaf*)lleaves->UncheckedAt(i);
         if (!leaf->GetBranch()) continue;
         sprintf(branchname,"%s.%s",leaf->GetBranch()->GetName(),leaf->GetName());
         // do not look at the indexes if any
         char *branch_dim = (char*)strstr(branchname,"[");
         if (branch_dim) { branch_dim[0] = '\0'; branch_dim++; }
         if (!strcmp(lname,branchname)) {
            TMethodCall *method = 0;
            fMethods.Add(method);
            Int_t code = fNcodes;
            fCodes[code] = i;
            fNcodes++;
            if (scanindex) fIndex[code] = index;
            else           fIndex[code] = -1;

            // Let see if we can understand the structure of this branch.
            // Usually we have: leafname[fixed_array] leaftitle[var_array]\type
            // (with fixed_array that can be a multi-dimension array.
            // NOTE I removed this? if ( !strcmp(lname,leaf->GetName() ) ) {
            const char *tname = leaf->GetTitle();
            char *dim = (char*)strstr( tname, "[" );
            Int_t virt_dim = 0;
            if (dim) {
               dim++;
               if (!branch_dim || strncmp(branch_dim,dim,strlen(branch_dim))) {
                  // then both are NOT the same so do the title first:
                  DefineDimensions( dim, code, virt_dim);
               }
            }
            if (branch_dim) {
               // then both are NOT same so do the title:
               DefineDimensions( branch_dim, code, virt_dim);
            }
            // should we also check the leaf name?

            if (leaf->InheritsFrom("TLeafC")) return 5000+code;
            return code;
         }
      }
//          Look for a member function
      for (i=0;i<nleaves;i++) {
         lname[dot] = 0;
         TLeaf *leaf = (TLeaf*)lleaves->UncheckedAt(i);
         if (!leaf->GetBranch()) continue;
         strcpy(branchname,leaf->GetBranch()->GetName());
         // do not look at the indexes if any
         char *branch_dim = (char*)strstr(branchname,"[");
         if (branch_dim) { branch_dim[0] = '\0'; branch_dim++; }
         if (!strcmp(lname,branchname)) {
            if (leaf->IsA() != TLeafObject::Class()) return -1;
            TLeafObject *lobj = (TLeafObject*)leaf;
            TMethodCall *method = lobj->GetMethodCall(lmethod);
            if (!method) return -1;
            fMethods.Add(method);
            Int_t code = fNcodes;
            fCodes[code] = i;
            fNcodes++;
            fIndex[code] = code-kMETHOD;

            // Let see if we can understand the structure of this branch.
            // Usually we have: leafname[fixed_array] leaftitle[var_array]\type
            // (with fixed_array that can be a multi-dimension array.
            // NOTE I removed this? if ( !strcmp(lname,leaf->GetName() ) ) {
            const char *tname = leaf->GetTitle();
            char *dim = (char*)strstr( tname, "[" );
            Int_t virt_dim = 0;
            if (dim) {
               dim++;
               if (!branch_dim || strncmp(branch_dim,dim,strlen(branch_dim))) {
                  // then both are NOT the same so do the title first:
                  DefineDimensions( dim, code, virt_dim);
               }
            }
            if (branch_dim) {
               // then both are NOT same so do the title:
               DefineDimensions( branch_dim, code, virt_dim);
            }
            // should we also check the leaf name?

            return code;
         }
      }

//*-*- There is no dot. Look for a data member only
   } else {
      for (i=0;i<nleaves;i++) {
         TLeaf *leaf = (TLeaf*)lleaves->UncheckedAt(i);
         if (!leaf->GetBranch()) continue;
         strcpy(branchname,leaf->GetBranch()->GetName());
         strcpy(leafname,leaf->GetName());
         // do not look at the indexes if any
         char *branch_dim = (char*)strstr(branchname,"[");
         if (branch_dim) { branch_dim[0] = '\0'; branch_dim++; }
         char *leaf_dim = (char*)strstr(leafname,"[");
         if (leaf_dim) { leaf_dim[0] = '\0'; leaf_dim++; }
         if (!strcmp(lname,leaf->GetBranch()->GetName()) ||
             !strcmp(lname,leaf->GetName())) {
            TMethodCall *method = 0;
            fMethods.Add(method);
            Int_t code = fNcodes;
            fCodes[code] = i;
            fNcodes++;
            if (scanindex) fIndex[code] = index;
            else           fIndex[code] = -1;

            // Let see if we can understand the structure of this branch.
            // Usually we have: leafname[fixed_array] leaftitle[var_array]\type
            // (with fixed_array that can be a multi-dimension array.
            // NOTE I removed this? if ( !strcmp(lname,leaf->GetName() ) ) {
            const char *tname = leaf->GetTitle();
            char *dim = (char*)strstr( tname, "[" );
            Int_t virt_dim = 0;
            if (dim) {
               dim++;
               if (!branch_dim || strncmp(branch_dim,dim,strlen(branch_dim))) {
                  // then both are NOT the same so do the title first:
                  DefineDimensions( dim, code, virt_dim);
               }
            }
            if (branch_dim) {
               // then both are NOT same so do the title:
               DefineDimensions( branch_dim, code, virt_dim);
            }
            // should we also check the leaf name?

            if (ref) printf("Cannot process reference to array index=%s[%s]\n",lname,ref);
            if (leaf->InheritsFrom("TLeafC") && !leaf->IsUnsigned()) return 5000+code;
            if (leaf->InheritsFrom("TLeafB") && !leaf->IsUnsigned()) return 5000+code;
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

  const Int_t kMAXSTRINGFOUND = 10;
  Int_t i,pos,pos2,int1,int2,real_instance,virt_dim;
  Float_t aresult;
  Double_t tab[kMAXFOUND];
  Double_t param[kMAXFOUND];
  Double_t dexp;
  char *tab2[kMAXSTRINGFOUND];

  if (fNoper == 1 && fNcodes > 0) {
     if (fCodes[0] < 0) {
        TCutG *gcut = (TCutG*)fMethods.At(0);
        TTreeFormula *fx = (TTreeFormula *)gcut->GetObjectX();
        TTreeFormula *fy = (TTreeFormula *)gcut->GetObjectY();
        Double_t xcut = fx->EvalInstance(instance);
        Double_t ycut = fy->EvalInstance(instance);
        return gcut->IsInside(xcut,ycut);
     }
     TLeaf *leaf = GetLeaf(0);

     // Now let calculate what physical instance we really need.
     // Some redundant code is used to speed up the cases where
     // they are no dimensions.
     real_instance = 0;
     Int_t max_dim = fNdimensions[0];
     if ( max_dim ) {
        virt_dim = 0;
        max_dim--;
        for (Int_t dim = 0; dim < max_dim; dim++) {
           if (fIndexes[0][dim]>=0) {
              real_instance += fIndexes[0][dim] * fCumulSize[0][dim+1];
           } else {
              if (fCumulUsedSize[virt_dim]>1) {
                 real_instance += ( ( instance % fCumulUsedSize[virt_dim] )
                                    / fCumulUsedSize[virt_dim+1])
                                  * fCumulSize[0][dim+1];
              } else {
                 real_instance += ( instance / fCumulUsedSize[virt_dim+1])
                                  * fCumulSize[0][dim+1];
              }
              virt_dim ++;
           }
        }
        if (fIndexes[0][max_dim]>=0) {
           real_instance += fIndexes[0][max_dim];
        } else {
           if (fCumulUsedSize[virt_dim]>1) {
              real_instance += instance % fCumulUsedSize[virt_dim];
           } else {
              real_instance += instance;
           }
        }
     }

     if (instance) {
        if (real_instance < leaf->GetNdata()) return leaf->GetValue(real_instance);
        else                                  return leaf->GetValue(0);
     } else {
        leaf->GetBranch()->GetEntry(fTree->GetReadEntry());
        if (!(leaf->IsA() == TLeafObject::Class())) return leaf->GetValue(real_instance);
        return GetValueLeafObject(fIndex[0],(TLeafObject *)leaf);
     }
  }
  for(i=0;i<fNval;i++) {
     if (fCodes[i] < 0) {
        TCutG *gcut = (TCutG*)fMethods.At(i);
        TTreeFormula *fx = (TTreeFormula *)gcut->GetObjectX();
        TTreeFormula *fy = (TTreeFormula *)gcut->GetObjectY();
        Double_t xcut = fx->EvalInstance(instance);
        Double_t ycut = fy->EvalInstance(instance);
        param[i] = gcut->IsInside(xcut,ycut);
     } else {
        TLeaf *leaf = GetLeaf(i);

        // Now let calculate what physical instance we really need.
        // Some redundant code is used to speed up the cases where
        // they are no dimensions.
        real_instance = 0;
        Int_t max_dim = fNdimensions[i];
        if ( max_dim ) {
           virt_dim = 0;
           max_dim--;
           for (Int_t dim = 0; dim < max_dim; dim++) {
              if (fIndexes[i][dim]>=0) {
                 real_instance += fIndexes[i][dim] * fCumulSize[i][dim+1];
              } else {
                 if (fCumulUsedSize[virt_dim]>1) {
                    real_instance += ( ( instance % fCumulUsedSize[virt_dim] )
                                       / fCumulUsedSize[virt_dim+1])
                                     * fCumulSize[i][dim+1];
                 } else {
                    real_instance += ( instance / fCumulUsedSize[virt_dim+1])
                                     * fCumulSize[i][dim+1];
                 }
                 virt_dim ++;
              }
           }
           if (fIndexes[i][max_dim]>=0) {
              real_instance += fIndexes[i][max_dim];
           } else {
              if (fCumulUsedSize[virt_dim]>1) {
                 real_instance += instance % fCumulUsedSize[virt_dim];
              } else {
                 real_instance += instance;
              }
           }
        }
        if (instance) {
           if (real_instance < leaf->GetNdata()) param[i] = leaf->GetValue(real_instance);
           else                                  param[i] = leaf->GetValue(0);
        } else {
           leaf->GetBranch()->GetEntry(fTree->GetReadEntry());
           if (!(leaf->IsA() == TLeafObject::Class())) param[i] = leaf->GetValue(real_instance);
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
          case  50 : pos++; tab[pos-1] = gRandom->Rndm(1); break;
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
//*-*-*-*-*-*-*-*Changed to Return number of available instances in the formula*-*-*-*-*-*-*-*
//*-*            =======================================
//

   // new version of GetNData:
   // Possible problem: we only allow one variable dimension so far.
   if (fMultiplicity==0) return 1;

   if (fMultiplicity==2) return fCumulUsedSize[0];

   // We have at least one leaf with a variable size:
   Int_t  overall, current;

   overall = 1;
   for (Int_t i=0;i<fNcodes;i++) {
      if (fCodes[i] < 0) continue;
      TLeaf *leaf = GetLeaf(i);
      if (leaf->GetLeafCount()) {
         TBranch *branch = leaf->GetLeafCount()->GetBranch();
         branch->GetEntry(fTree->GetReadEntry());
         current = leaf->GetLen() / leaf->GetLenStatic();
         if (fIndexes[i][0] < 0 ) {
            if (overall==1 || (current!=1 && current<overall) ) overall = current;
         } else if (fIndexes[i][0] >= current) {
            // unreacheable element requested:
            overall = 0;
         }
      }
   }
   if (overall==0) return 0;
   if (fCumulUsedSize[0] >= 0 ) return fCumulUsedSize[0];
   else {
      Int_t tmp = -1*fCumulUsedSize[0];
      if (tmp!=1 && (tmp<overall) ) return tmp*fCumulUsedSize[1];
      return overall*fCumulUsedSize[1];
   }

}

//______________________________________________________________________________
Double_t TTreeFormula::GetValueLeafObject(Int_t i, TLeafObject *leaf)
{
//*-*-*-*-*-*-*-*Return result of a leafobject method*-*-*-*-*-*-*-*
//*-*            ====================================
//

   if (i>=0) return 0; // case where we do NOT have a method defined
   TMethodCall *m = GetMethodCall(i);
   if (!m)   return 0;

   void *thisobj = leaf->GetObject();

   TMethodCall::EReturnType r = m->ReturnType();

   if (r == TMethodCall::kLong) {
      Long_t l;
      m->Execute(thisobj, l);
      return (Double_t) l;
   }
   if (r == TMethodCall::kDouble) {
      Double_t d;
      m->Execute(thisobj, d);
      return (Double_t) d;
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
         sprintf(value,"%9.9g",EvalInstance(0));
         char *expo = strchr(value,'e');
         if (expo) {
            if (value[0] == '-') strcpy(expo-6,expo);            
            else                 strcpy(expo-5,expo);            
         }
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
