// @(#)root/ldap:$Id$
// Author: Evgenia Smirnova   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLDAPAttribute.h"
#include "TObjString.h"
#include "Riostream.h"


ClassImp(TLDAPAttribute)

////////////////////////////////////////////////////////////////////////////////
///constructor

TLDAPAttribute::TLDAPAttribute(const char *name) : fNCount(0)
{
   SetName(name);
   fValues = new TList;
   fValues->SetOwner();
}

////////////////////////////////////////////////////////////////////////////////
/// Creates an Attribute with name and value.

TLDAPAttribute::TLDAPAttribute(const char *name, const char *value)
   : fNCount(0)
{
   SetName(name);
   fValues = new TList;
   fValues->SetOwner();
   AddValue(value);
}

////////////////////////////////////////////////////////////////////////////////
/// LDAP attribute copy ctor.

TLDAPAttribute::TLDAPAttribute(const TLDAPAttribute &attr)
   : TNamed(attr), fNCount(attr.fNCount)
{
   fValues = new TList;
   fValues->SetOwner();

   TIter next(attr.fValues);
   while (TObjString *str = (TObjString*) next()) {
      fValues->AddLast(new TObjString(str->GetName()));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Equal operator

TLDAPAttribute& TLDAPAttribute::operator=(const TLDAPAttribute &attr)
{
   if(this!=&attr) {
      TNamed::operator=(attr);
      fValues=attr.fValues;
      fNCount=attr.fNCount;
   } return *this;
}

////////////////////////////////////////////////////////////////////////////////
///destructor

TLDAPAttribute::~TLDAPAttribute()
{
   delete fValues;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a value to the attribute.

void TLDAPAttribute::AddValue(const char *value)
{
   fValues->AddLast(new TObjString(value));
}

////////////////////////////////////////////////////////////////////////////////
/// Delete value by name.

void TLDAPAttribute::DeleteValue(const char *value)
{
   Int_t n = GetCount();
   for (Int_t i = 0; i < n; i++) {
      TObjString *v = (TObjString*) fValues->At(i);
      if (v->String().CompareTo(value) == 0) {
         delete fValues->Remove(v);
         if (fNCount > i) fNCount--;
         return;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get next value of the attribute. Returns zero after the last value,
/// then returns the first value again.

const char *TLDAPAttribute::GetValue() const
{
   Int_t n = GetCount();
   if (n > fNCount) {
      return ((TObjString*)fValues->At(fNCount++))->GetName();
   } else {
      fNCount = 0;
      return 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Print an attribute.

void TLDAPAttribute::Print(Option_t *) const
{
   Int_t counter = GetCount();
   if (counter == 0) {
      std::cout << GetName() << ": " << std::endl;
   } else if (counter != 0) {
      for (Int_t i = 0; i < counter; i++) {
         std::cout << GetName() << ": " << GetValue() << std::endl;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get "LDAPMod" structure for attribute. Returned LDAPMod must be
/// deleted by the user.

LDAPMod *TLDAPAttribute::GetMod(Int_t op)
{
   LDAPMod *tmpMod = new LDAPMod;
   Int_t iCount = GetCount();
   char **values = new char* [iCount + 1];
   char *type = new char [strlen(GetName())+1];
   for (int i = 0; i < iCount; i++) {
      int nch = strlen(((TObjString*)fValues->At(i))->GetName()) + 1;
      values[i] = new char [nch];
      strlcpy(values[i], ((TObjString*)fValues->At(i))->GetName(),nch);
   }

   values[iCount] = 0;
   strlcpy(type, GetName(),strlen(GetName())+1);
   tmpMod->mod_values = values;
   tmpMod->mod_type = type;
   tmpMod->mod_op = op;

   return tmpMod;
}
