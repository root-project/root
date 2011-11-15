//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ProcFileElements                                                     //
//                                                                      //
// This class holds information about the processed elements of a file. //
// Used for testing.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "ProcFileElements.h"
#include "TCollection.h"

//_______________________________________________________________________
Int_t ProcFileElements::ProcFileElement::Compare(const TObject *o) const
{
   // Compare this element with 'e'.
   // Return
   //          -1      this should come first
   //           0      same
   //           1      e should come first

   const ProcFileElements::ProcFileElement *e =
      dynamic_cast<const ProcFileElements::ProcFileElement *>(o);
   if (!e) return -1;

   if (fFirst == e->fFirst) {
      // They start at the same point 
      return 0;
   } else if (fFirst < e->fFirst) {
      // This starts first
      return -1;
   } else {
      // e starts first
      return 1;
   }
}

//_______________________________________________________________________
Int_t ProcFileElements::ProcFileElement::Overlapping(ProcFileElement *e)
{
   // Check overlapping status of this element with 'e'.
   // Return
   //          -1      not overlapping
   //           0      adjacent
   //           1      overlapping

   if (!e) return -1;

   if (fFirst == 0 && fLast == -1) {
      // We cover the full range, so we overlap
      return 1;
   }

   if (fFirst < e->fFirst) {
      if (fLast >= 0) {
         if (fLast < e->fFirst - 1) {
            // Disjoint
            return -1;
         } else {
            // We somehow overlap
            if (fLast == e->fFirst - 1) {
               // Just adjacent
               return 0;
            } else {
               // Real overlap
               return 1;
            }
         }
      } else {
         // Always overlapping
         return 1;
      }
   } else if (fFirst == e->fFirst) {
      // Overlapping or illy adjacent
      if (fFirst == fLast || e->fFirst == e->fLast) return 0;
      return 1;
   } else {
      // The other way around
      if (e->fLast >= 0) {
         if (e->fLast < fFirst - 1) {
            // Disjoint
            return -1;
         } else {
            // We somehow overlap
            if (e->fLast == fFirst - 1) {
               // Just adjacent
               return 0;
            } else {
               // Real overlap
               return 1;
            }
         }
      } else {
         // Always overlapping
         return 1;
      }
   }

   // Should never be here
   Warning("Overlapping", "should never be here!");
   return -1;
}

//_______________________________________________________________________
Int_t ProcFileElements::ProcFileElement::MergeElement(ProcFileElement *e)
{
   // Merge this element with element 'e'.
   // Return -1 if it cannot be done, i.e. thei are disjoint; 0 otherwise

   // Check if it can be done
   if (Overlapping(e) < 0) return -1;
   
   // Ok, we can merge: set the lower bound
   if (e->fFirst < fFirst) fFirst = e->fFirst;
   
   // Set the upper bound
   if (fLast == -1 || e->fLast == -1) {
      fLast = -1;
   } else {
      if (fLast < e->fLast) fLast = e->fLast;
   }
   // Done
   return 0;
}

//_______________________________________________________________________
void ProcFileElements::ProcFileElement::Print(Option_t *) const
{
   // Print range of this element

   Printf("\tfirst: %lld\t last: %lld", fFirst, fLast);
}

//_______________________________________________________________________
Int_t ProcFileElements::Add(Long64_t fst, Long64_t lst)
{
   // Add a new element to the list 
   // Return 1 if a new element has been added, 0 if it has been merged
   // with an existing one, -1 in case of error

   if (!fElements) fElements = new TSortedList;
   if (!fElements) {
      Error("Add", "could not create internal list!");
      return -1;
   }
   
   // Create (tempoarry element)
   ProcFileElements::ProcFileElement *ne =
      new ProcFileElements::ProcFileElement(fst, lst);

   // Check if if it is adjacent or overalapping with an existing one
   TIter nxe(fElements);
   ProcFileElements::ProcFileElement *e = 0;
   while ((e = (ProcFileElements::ProcFileElement *)nxe())) {
      if (e->MergeElement(ne) == 0) break;
   }
   
   Int_t rc = 0;
   // Remove and re-add the merged element to sort correctly its possibly new position
   if (e) {
      fElements->Remove(e);
      fElements->Add(e);
      SafeDelete(ne);
   } else {
      // Add the new element
      fElements->Add(ne);
      rc = 1;
   }

   // New overall ranges
   if (fElements) {
      if ((e = (ProcFileElements::ProcFileElement *) fElements->First())) fFirst = e->fFirst;
      if ((e = (ProcFileElements::ProcFileElement *) fElements->Last())) fLast = e->fLast;
   }
   
   // Done
   return rc;
}

//_______________________________________________________________________
void ProcFileElements::Print(Option_t *) const
{
   // Print info about this processed file

   Printf("--- ProcFileElements ----------------------------------------");
   Printf(" File: %s", fName.Data());
   Printf(" # proc elements: %d", fElements ? fElements->GetSize() : 0);
   TIter nxe(fElements);
   ProcFileElements::ProcFileElement *e = 0;
   while ((e = (ProcFileElements::ProcFileElement *)nxe())) { e->Print(); }
   Printf(" Raw overall range: [%lld, %lld]", fFirst, fLast);
   Printf("-------------------------------------------------------------");
}

//_______________________________________________________________________
Int_t ProcFileElements::Merge(TCollection *li)
{
   // Merge this object with those in the list
   // Return number of elements added

   if (!li) return -1;
   
   if (li->GetSize() <= 0) return 0;

   Int_t nadd = 0;
   TIter nxo(li);
   ProcFileElements *pfe = 0;
   while ((pfe = (ProcFileElements *) nxo())) {
      if (strcmp(GetName(), pfe->GetName()))
         Warning("Merge", "merging objects of different name! ('%s' != '%s')",
                          GetName(),  pfe->GetName());
      TIter nxe(pfe->GetListOfElements());
      ProcFileElements::ProcFileElement *e = 0;
      while ((e = (ProcFileElements::ProcFileElement *)nxe())) {
         Int_t rc = Add(e->fFirst, e->fLast);
         if (rc == 1) nadd++;
      }
   }
   // Done
   return nadd;
}
