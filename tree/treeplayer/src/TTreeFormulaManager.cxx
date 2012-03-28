// @(#)root/treeplayer:$Id$
// Author: Philippe Canal   20/03/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TTreeFormulaManager.h"

#include "TArrayI.h"
#include "TError.h"
#include "TLeafElement.h"


ClassImp(TTreeFormulaManager)

//________________________________________________________________________________
//
//     A TreeFormulaManager is used to coordinate one or more TTreeFormula objecs
//
//  In particular it makes sure that the dimensions and size of all the formulas
//  is properly coordinated.
//


//______________________________________________________________________________
TTreeFormulaManager::TTreeFormulaManager() : TObject()
{
//*-*-*-*-*-*-*-*-*-*-*Tree FormulaManger default constructor*-*-*-*-*-*-*-*-*-*
//*-*                  ======================================

   fMultiplicity = 0;
   fMultiVarDim  = kFALSE;
   fNeedSync     = kFALSE;
   fNdata = 1;

   for(Int_t i=0; i<kMAXFORMDIM+1; i++) {
      fVarDims[i] = 0;
      fCumulUsedSizes[i] = 1;
      fUsedSizes[i] = 1;
      fVirtUsedSizes[i] = 1;
   }
   fCumulUsedVarDims = 0;
}


//______________________________________________________________________________
TTreeFormulaManager::~TTreeFormulaManager()
{
//*-*-*-*-*-*-*-*-*-*-*Tree FormulaManager default destructor*-*-*-*-*-*-*-*-*-*
//*-*                  ======================================

   for (int l = 0; l<kMAXFORMDIM; l++) {
      if (fVarDims[l]) delete fVarDims[l];
      fVarDims[l] = 0;
   }
   if (fCumulUsedVarDims) delete fCumulUsedVarDims;
}

//______________________________________________________________________________
void TTreeFormulaManager::Remove(TTreeFormula* adding)
{
   // Remove a formula from this manager

   fFormulas.Remove(adding);
}

//______________________________________________________________________________
void TTreeFormulaManager::Add(TTreeFormula* adding)
{
  // Add a new formula to the list of formulas managed
  // The manager of the formula will be changed and the old one will be deleted
  // if it is empty.

   TTreeFormulaManager * old = adding->fManager;

   if (old) {
      if (old==this) {
         if (fFormulas.FindObject(adding)) return;
      } else {
         old->fFormulas.Remove(adding);
         if (old->fFormulas.GetLast()==-1) delete adding->fManager;
      }
   }

   if (adding->TestBit(TTreeFormula::kNeedEntries)) {
      SetBit(TTreeFormula::kNeedEntries);
   }

   fFormulas.Add(adding);
   adding->fManager = this;
   fNeedSync = kTRUE;
}

//______________________________________________________________________________
void TTreeFormulaManager::AddVarDims(Int_t virt_dim)
{
   // Add a variable dimension

   if (!fVarDims[virt_dim]) fVarDims[virt_dim] = new TArrayI;
}

//______________________________________________________________________________
void TTreeFormulaManager::CancelDimension(Int_t virt_dim)
{
   // Cancel a dimension.  This is usually called when an out-of-bounds index
   // is used.

   fCumulUsedSizes[virt_dim] = 0;

}

//______________________________________________________________________________
void TTreeFormulaManager::EnableMultiVarDims()
{
   // Set the manager as handling a formula with multiple variable dimensions

   fMultiVarDim = kTRUE;
   if (!fCumulUsedVarDims) fCumulUsedVarDims = new TArrayI;

}

//______________________________________________________________________________
Int_t TTreeFormulaManager::GetNdata(Bool_t forceLoadDim)
{
//*-*-*-*-*-*-*-*Return number of available instances in the formulas*-*-*-*-*-*
//*-*            ====================================================
//

   Int_t k;

   // new version of GetNData:
   // Possible problem: we only allow one variable dimension so far.
   if (fMultiplicity==0) return fNdata;

   if (fMultiplicity==2) return fNdata; // CumulUsedSizes[0];

   // We have at least one leaf with a variable size:

   // Reset the registers.
   for(k=0; k<=kMAXFORMDIM; k++) {
      fUsedSizes[k] = TMath::Abs(fVirtUsedSizes[k]);
      if (fVarDims[k]) {
         for(Int_t i0=0;i0<fVarDims[k]->GetSize();i0++) {
            fVarDims[k]->AddAt(0,i0);
         }
      }
   }
   if (fCumulUsedVarDims) {
      for(Int_t i0=0;i0<fCumulUsedVarDims->GetSize();++i0) {
         fCumulUsedVarDims->AddAt(0,i0);
      }
   }

   TTreeFormula* current = 0;

   Int_t size = fFormulas.GetLast()+1;

   for(Int_t i=0; i<size; i++) {

      current = (TTreeFormula*)fFormulas.UncheckedAt(i);
      if (current->fMultiplicity!=1 && !current->fHasCast) continue;
      if (!current->LoadCurrentDim() ) {
         if (forceLoadDim) {
            for(Int_t j=i+1; j<size; j++) {
               current = (TTreeFormula*)fFormulas.UncheckedAt(j);
               if (current->fMultiplicity!=1 && !current->fHasCast) continue;
               current->LoadCurrentDim();
            }
         }
         fNdata = 0;
         return 0;
      }
   }

   if (fMultiplicity==-1) { fNdata = 1; return fCumulUsedSizes[0]; }

   Int_t overall = 1;
   if (!fMultiVarDim) {
      for (k = kMAXFORMDIM; (k >= 0) ; k--) {
         if (fUsedSizes[k]>=0) {
            overall *= fUsedSizes[k];
            fCumulUsedSizes[k] = overall;
         } else {
            Error("GetNdata","a dimension is still negative!");
         }
      }
   } else {
      overall = 0; // Since we work with additions in this section
      if (fCumulUsedVarDims && fUsedSizes[0]>fCumulUsedVarDims->GetSize()) fCumulUsedVarDims->Set(fUsedSizes[0]);
      for(Int_t i = 0; i < fUsedSizes[0]; i++) {
         Int_t local_overall = 1;
         for (k = kMAXFORMDIM; (k > 0) ; k--) {
            if (fVarDims[k]) {
               Int_t index = fVarDims[k]->At(i);
               if (fCumulUsedVarDims && fCumulUsedVarDims->At(i)==1 && index) index = 1;
               if (fUsedSizes[k]==1 || (index!=1 && index<fUsedSizes[k]))
                  local_overall *= index;
               else local_overall *= fUsedSizes[k];
            } else {
               local_overall *= fUsedSizes[k];
            }
         }
         // a negative value indicates that this value of the primary index
         // will lead to an invalid index; So we skip it.
         if (fCumulUsedVarDims->At(i)<0) fCumulUsedVarDims->AddAt(0,i);
         else {
            fCumulUsedVarDims->AddAt(local_overall,i);
            overall += local_overall;
         }
      }
   }
   fNdata = overall;
   return overall;

}

//______________________________________________________________________________
Bool_t TTreeFormulaManager::Sync()
{
   // Synchronize all the formulae.

   if (!fNeedSync) return true;

   TTreeFormula* current = 0;
   Bool_t hasCast = kFALSE;

   fMultiplicity = 0;
   // We do not use an intermediary variable because ResetDimensions
   // might add more formulas (TCutG).
   for(Int_t i=0; i<fFormulas.GetLast()+1; i++) {
      current = (TTreeFormula*)fFormulas.UncheckedAt(i);

      hasCast |= current->fHasCast;

      // We probably need to reset the formula's dimension

      current->ResetDimensions();
      switch (current->GetMultiplicity()) {
         case 0:
            // nothing to do
            break;
         case 1:
            fMultiplicity = 1;
            break;
         case 2:
            if (fMultiplicity!=1) fMultiplicity = 2;
            break;
         default:
            Error("Sync","Unexpected case!");
      }


   } // end of for each formulas

   // For now we keep fCumulUsedSizes sign aware.
   // This will be reset properly (if needed) by GetNdata.
   fCumulUsedSizes[kMAXFORMDIM] = fUsedSizes[kMAXFORMDIM];
   for (Int_t k = kMAXFORMDIM; (k > 0) ; k--) {
      if (fUsedSizes[k-1]>=0) {
         fCumulUsedSizes[k-1] = fUsedSizes[k-1] * fCumulUsedSizes[k];
      } else {
         fCumulUsedSizes[k-1] = - TMath::Abs(fCumulUsedSizes[k]);
      }
   }

   // Now that we know the virtual dimension we know if a loop over EvalInstance
   // is needed or not.
   if (fCumulUsedSizes[0]==1 && fMultiplicity>0) {
      // Case where even though we have an array.  We know that they will always
      // only be one element.
      fMultiplicity -= 2;
   } else if (fCumulUsedSizes[0]<0 && fMultiplicity==2) {
      // Case of a fixed length array that have one of its indices given
      // by a variable.
      fMultiplicity = 1;
   } else if (fMultiplicity==0 && hasCast) {
      fMultiplicity = -1;
   }

   switch(fMultiplicity) {
      case 0: fNdata = 1; break;
      case 2: fNdata = fCumulUsedSizes[0]; break;
      default: fNdata = 0;
   }
   fNeedSync = kFALSE;

   return true;
}

//______________________________________________________________________________
void TTreeFormulaManager::UpdateFormulaLeaves()
{
   // this function could be called TTreePlayer::UpdateFormulaLeaves, itself
   // called by TChain::LoadTree when a new Tree is loaded.
   // Because Trees in a TChain may have a different list of leaves, one
   // must update the leaves numbers in the TTreeFormula used by the TreePlayer.

   // A safer alternative would be to recompile the whole thing .... However
   // currently compile HAS TO be called from the constructor!

   Int_t size = fFormulas.GetLast()+1;

   for(Int_t i=0; i<size; i++) {

      TTreeFormula *current = (TTreeFormula*)fFormulas.UncheckedAt(i);
      current->UpdateFormulaLeaves();

   }

}

//______________________________________________________________________________
void TTreeFormulaManager::UpdateUsedSize(Int_t &virt_dim, Int_t vsize)
{
   // Reload the array sizes

   if (vsize<0)
      fVirtUsedSizes[virt_dim] = -1 * TMath::Abs(fVirtUsedSizes[virt_dim]);
   else
      if ( TMath::Abs(fVirtUsedSizes[virt_dim])==1
          || (vsize < TMath::Abs(fVirtUsedSizes[virt_dim]) ) ) {
         // Absolute values represent the min of all real dimensions
         // that are known.  The fact that it is negatif indicates
         // that one of the leaf has a variable size for this
         // dimensions.
         if (fVirtUsedSizes[virt_dim] < 0) {
            fVirtUsedSizes[virt_dim] = -1 * vsize;
         } else {
            fVirtUsedSizes[virt_dim] = vsize;
         }
      }
   fUsedSizes[virt_dim] = fVirtUsedSizes[virt_dim];
   virt_dim++;

}
