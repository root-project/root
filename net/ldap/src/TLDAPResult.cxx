// @(#)root/ldap:$Id$
// Author: Oleksandr Grebenyuk   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLDAPResult.h"
#include "TLDAPEntry.h"
#include "TLDAPAttribute.h"


ClassImp(TLDAPResult)

//______________________________________________________________________________
TLDAPResult::TLDAPResult(LDAP *ld, LDAPMessage *searchresult)
   : fLd(ld), fSearchResult(searchresult), fCurrentEntry(searchresult)
{
   // TLDAPResult object is just a wrapper of the LDAPMessage structure.
   // LDAP *ld:                  The current session handler
   // LDAPMessage *searchresult: The LDAPMessage structure returned from
   //                            the ldap_search_s() call

   if (!GetCount())
      fCurrentEntry = 0;
}

//______________________________________________________________________________
TLDAPResult::TLDAPResult(const TLDAPResult& ldr) :
   TObject(ldr),
   fLd(ldr.fLd),
   fSearchResult(ldr.fSearchResult),
   fCurrentEntry(ldr.fCurrentEntry)
{
   // Copy constructor
}

//______________________________________________________________________________
TLDAPResult& TLDAPResult::operator=(const TLDAPResult& ldr)
{
   // Equal operator
   if(this!=&ldr) {
      TObject::operator=(ldr);
      fLd=ldr.fLd;
      fSearchResult=ldr.fSearchResult;
      fCurrentEntry=ldr.fCurrentEntry;
   } return *this;
}

//______________________________________________________________________________
TLDAPResult::~TLDAPResult()
{
   // Deletes the LDAPMessage structure

   if (fSearchResult)
      ldap_msgfree(fSearchResult);
}

//______________________________________________________________________________
TLDAPEntry *TLDAPResult::GetNext()
{
   // Returns next entry from the search result.
   // After the last entry it returns a zero pointer
   // and after this it returns the first entry again.
   // The user is responsable for deleting the returned object after use.

   TLDAPEntry *entry = CreateEntry(fCurrentEntry);
   fCurrentEntry = (fCurrentEntry != 0 ? ldap_next_entry(fLd, fCurrentEntry) :
                   (GetCount() != 0 ? fSearchResult : 0));
   return entry;
}

//______________________________________________________________________________
TLDAPEntry *TLDAPResult::CreateEntry(LDAPMessage *entry)
{
   // Creates TLDAPEntry object from the data containing in the LDAPMessage
   // structure and returns pointer to it.
   // The user is responsable for deleting the returned object after use.
   // LDAPMessage *entry: Pointer to the LDAPMessage structure containing
   // the entry data.

   if (entry == 0)
      return 0;

   char *dn;
   char *attr;
   BerValue   **vals;
   BerElement *ptr;

   dn = ldap_get_dn(fLd, entry);
   TLDAPEntry *ldapentry = new TLDAPEntry(dn);
   for (attr = ldap_first_attribute(fLd, entry, &ptr); attr != 0;
        attr = ldap_next_attribute(fLd, entry, ptr)) {
      TLDAPAttribute attribute(attr);
      vals = ldap_get_values_len(fLd, entry, attr);
      if (vals) {
         for (Int_t i = 0; vals[i] != 0; i++) {
            attribute.AddValue(vals[i]->bv_val);
         }
         ldap_value_free_len(vals);
      }
      ldapentry->AddAttribute(attribute);
   }

   return ldapentry;
}

//______________________________________________________________________________
Int_t TLDAPResult::GetCount() const
{
   // Returns the number of entries in the search result

   LDAP *ld = fLd;
   LDAPMessage *result = fSearchResult;

   return ldap_count_entries(ld, result);
}

//______________________________________________________________________________
void TLDAPResult::Print(Option_t *) const
{
   // Prints all entries.
   // Calls the Print() member function of the each entry.

   TLDAPEntry *e;
   Int_t count = GetCount() + 1;
   for (Int_t i = 0; i < count; i++) {
      e = const_cast<TLDAPResult*>(this)->GetNext();
      if (e) {
         e->Print();
         delete e;
      }
   }
}
