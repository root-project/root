// @(#)root/ldap:$Id$
// Author: Evgenia Smirnova   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLDAPEntry.h"
#include "TLDAPAttribute.h"
#include "Riostream.h"


ClassImp(TLDAPEntry)

//______________________________________________________________________________
TLDAPEntry::TLDAPEntry(const char *dn) : fNCount(0)
{
   // Creates the new TLDAPEntry object with the specified DN (distinguished
   // name) and the empty list of attributes.
   // const char *dn: The DN of the entry. You can change it later by calling
   //                 the SetDn() member function

   SetDn(dn);
   fAttr = new TList;
   fAttr->SetOwner();
}

//______________________________________________________________________________
TLDAPEntry::TLDAPEntry(const TLDAPEntry &e) : TObject(e), fNCount(e.fNCount)
{
   // Copy ctor.

   SetDn(e.GetDn());
   fAttr = new TList;
   fAttr->SetOwner();

   TIter next(e.fAttr);
   while (TLDAPAttribute *att = (TLDAPAttribute *)next()) {
      fAttr->AddLast(new TLDAPAttribute(*att));
   }
}

//______________________________________________________________________________
TLDAPEntry& TLDAPEntry::operator=(const TLDAPEntry& lde) 
{
   // Equal operator
   if(this!=&lde) {
      TObject::operator=(lde);
      fDn=lde.fDn;
      fAttr=lde.fAttr;
      fNCount=lde.fNCount;
   } return *this;
}

//______________________________________________________________________________
TLDAPEntry::~TLDAPEntry()
{
   // Deletes all the attributes of the entry.

   delete fAttr;
}

//______________________________________________________________________________
void TLDAPEntry::AddAttribute(const TLDAPAttribute &attr)
{
   // Add an attribute to the entry.
   // TLDAPAtrribute attr: attribute to be added.

   fAttr->AddLast(new TLDAPAttribute(attr));
}

//______________________________________________________________________________
void TLDAPEntry::Print(Option_t *) const
{
   // Print entry in LDIF format.

   cout << "dn: "<< fDn << endl;
   TLDAPAttribute *attr = GetAttribute("objectClass");
   if (attr != 0)
      attr->Print();
   Int_t n = GetCount();
   for (Int_t i = 0; i < n; i++) {
      attr = (TLDAPAttribute*) fAttr->At(i);
      if (TString(attr->GetName()).CompareTo("objectClass", TString::kIgnoreCase) != 0)
         attr->Print();
   }
   cout << endl;
}

//______________________________________________________________________________
TLDAPAttribute *TLDAPEntry::GetAttribute() const
{
   // Get next attribute of the entry. Returns zero after the last attribute,
   // then returns the first attribute again.

   Int_t n = GetCount();
   if (n > fNCount) {
      return (TLDAPAttribute*)fAttr->At(fNCount++);
   } else {
      fNCount = 0;
      return 0;
   }
}

//______________________________________________________________________________
TLDAPAttribute *TLDAPEntry::GetAttribute(const char *name) const
{
   // Get attribute by name.
   // Doesn't affect the order of attributes to be returned from the
   // next GetAttribute() call. Attribute name is case insensitive.

   Int_t n = GetCount();
   for (Int_t i = 0; i < n; i++) {
      if (TString(((TLDAPAttribute*)fAttr->At(i))->GetName()).CompareTo(name, TString::kIgnoreCase) == 0) {
         return (TLDAPAttribute*)fAttr->At(i);
      }
   }
   return 0;
}

//______________________________________________________________________________
void TLDAPEntry::DeleteAttribute(const char *name)
{
   // Delete attribute by name.
   // Attribute name is case insensitive.

   Int_t n = GetCount();
   for (Int_t i = 0; i < n; i++) {
      if (TString(((TLDAPAttribute*)fAttr->At(i))->GetName()).CompareTo(name, TString::kIgnoreCase) == 0) {
         delete fAttr->Remove(fAttr->At(i));
         if (fNCount > i) fNCount--;
         return;
      }
   }
}

//______________________________________________________________________________
Bool_t TLDAPEntry::IsReferral() const
{
   // Check if entry is referal.

   Bool_t att = kFALSE;
   Bool_t obj = kFALSE;
   Int_t n = GetCount();
   TString name;
   for (Int_t i = 0; (i < n) && (!att || !obj); i++) {
      name = TString(((TLDAPAttribute*) fAttr->At(i))->GetName());
      if (name.CompareTo("ref", TString::kIgnoreCase) == 0) {
         att = kTRUE;
      } else {
         if (name.CompareTo("objectclass", TString::kIgnoreCase) == 0) {
            TLDAPAttribute *attr = (TLDAPAttribute*)fAttr->At(i);
            Int_t valcnt = attr->GetCount() + 1;
            for (Int_t j = 0; (j < valcnt) && (!obj); j++)
               obj |= (Bool_t)TString(attr->GetValue()).CompareTo("referral", TString::kIgnoreCase);
         }
      }
   }
   return (att && obj);
}

//______________________________________________________________________________
TList *TLDAPEntry::GetReferrals() const
{
   // Get the TList of referrals.
   // Returns an empty list if entry is not referral.
   // User is responsible for deleting returned TList.

   TList *list = new TList;
   TLDAPAttribute *ref = GetAttribute("ref");
   if (ref != 0) {
      Int_t n = ref->GetCount();
      for (Int_t i = 0; i < n; i++) {
         list->Add(ref->fValues->At(i));
      }
   }
   return list;
}

//______________________________________________________________________________
LDAPMod **TLDAPEntry::GetMods(Int_t op)
{
   // Get array of "LDAPMod" structures for entry.

   Int_t n = GetCount();
   LDAPMod **mods = new LDAPMod* [n + 1];
   for (Int_t i = 0; i < n; i++)
      mods[i] = ((TLDAPAttribute*)(fAttr->At(i)))->GetMod(op);
   mods[n] = 0;
   return mods;
}
