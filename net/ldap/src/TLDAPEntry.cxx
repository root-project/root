// @(#)root/ldap:$Id$
// Author: Evgenia Smirnova   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLDAPEntry.h"
#include "TLDAPAttribute.h"
#include "Riostream.h"


ClassImp(TLDAPEntry);

////////////////////////////////////////////////////////////////////////////////
/// Creates the new TLDAPEntry object with the specified DN (distinguished
/// name) and the empty list of attributes.
/// const char *dn: The DN of the entry. You can change it later by calling
///                 the SetDn() member function

TLDAPEntry::TLDAPEntry(const char *dn) : fNCount(0)
{
   SetDn(dn);
   fAttr = new TList;
   fAttr->SetOwner();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy ctor.

TLDAPEntry::TLDAPEntry(const TLDAPEntry &e) : TObject(e), fNCount(e.fNCount)
{
   SetDn(e.GetDn());
   fAttr = new TList;
   fAttr->SetOwner();

   TIter next(e.fAttr);
   while (TLDAPAttribute *att = (TLDAPAttribute *)next()) {
      fAttr->AddLast(new TLDAPAttribute(*att));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Equal operator

TLDAPEntry& TLDAPEntry::operator=(const TLDAPEntry& lde)
{
   if(this!=&lde) {
      TObject::operator=(lde);
      fDn=lde.fDn;
      fAttr=lde.fAttr;
      fNCount=lde.fNCount;
   } return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Deletes all the attributes of the entry.

TLDAPEntry::~TLDAPEntry()
{
   delete fAttr;
}

////////////////////////////////////////////////////////////////////////////////
/// Add an attribute to the entry.
/// TLDAPAtrribute attr: attribute to be added.

void TLDAPEntry::AddAttribute(const TLDAPAttribute &attr)
{
   fAttr->AddLast(new TLDAPAttribute(attr));
}

////////////////////////////////////////////////////////////////////////////////
/// Print entry in LDIF format.

void TLDAPEntry::Print(Option_t *) const
{
   std::cout << "dn: "<< fDn << std::endl;
   TLDAPAttribute *attr = GetAttribute("objectClass");
   if (attr != 0)
      attr->Print();
   Int_t n = GetCount();
   for (Int_t i = 0; i < n; i++) {
      attr = (TLDAPAttribute*) fAttr->At(i);
      if (TString(attr->GetName()).CompareTo("objectClass", TString::kIgnoreCase) != 0)
         attr->Print();
   }
   std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Get next attribute of the entry. Returns zero after the last attribute,
/// then returns the first attribute again.

TLDAPAttribute *TLDAPEntry::GetAttribute() const
{
   Int_t n = GetCount();
   if (n > fNCount) {
      return (TLDAPAttribute*)fAttr->At(fNCount++);
   } else {
      fNCount = 0;
      return 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get attribute by name.
/// Doesn't affect the order of attributes to be returned from the
/// next GetAttribute() call. Attribute name is case insensitive.

TLDAPAttribute *TLDAPEntry::GetAttribute(const char *name) const
{
   Int_t n = GetCount();
   for (Int_t i = 0; i < n; i++) {
      if (TString(((TLDAPAttribute*)fAttr->At(i))->GetName()).CompareTo(name, TString::kIgnoreCase) == 0) {
         return (TLDAPAttribute*)fAttr->At(i);
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete attribute by name.
/// Attribute name is case insensitive.

void TLDAPEntry::DeleteAttribute(const char *name)
{
   Int_t n = GetCount();
   for (Int_t i = 0; i < n; i++) {
      if (TString(((TLDAPAttribute*)fAttr->At(i))->GetName()).CompareTo(name, TString::kIgnoreCase) == 0) {
         delete fAttr->Remove(fAttr->At(i));
         if (fNCount > i) fNCount--;
         return;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if entry is referal.

Bool_t TLDAPEntry::IsReferral() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Get the TList of referrals.
/// Returns an empty list if entry is not referral.
/// User is responsible for deleting returned TList.

TList *TLDAPEntry::GetReferrals() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Get array of "LDAPMod" structures for entry.

LDAPMod **TLDAPEntry::GetMods(Int_t op)
{
   Int_t n = GetCount();
   LDAPMod **mods = new LDAPMod* [n + 1];
   for (Int_t i = 0; i < n; i++)
      mods[i] = ((TLDAPAttribute*)(fAttr->At(i)))->GetMod(op);
   mods[n] = 0;
   return mods;
}
