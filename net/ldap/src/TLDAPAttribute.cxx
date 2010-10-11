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

//______________________________________________________________________________
TLDAPAttribute::TLDAPAttribute(const char *name) : fNCount(0)
{
   //constructor
   SetName(name);
   fValues = new TList;
   fValues->SetOwner();
}

//______________________________________________________________________________
TLDAPAttribute::TLDAPAttribute(const char *name, const char *value)
   : fNCount(0)
{
   // Creates an Attribute with name and value.

   SetName(name);
   fValues = new TList;
   fValues->SetOwner();
   AddValue(value);
}

//______________________________________________________________________________
TLDAPAttribute::TLDAPAttribute(const TLDAPAttribute &attr)
   : TNamed(attr), fNCount(attr.fNCount)
{
   // LDAP attribute copy ctor.

   fValues = new TList;
   fValues->SetOwner();

   TIter next(attr.fValues);
   while (TObjString *str = (TObjString*) next()) {
      fValues->AddLast(new TObjString(str->GetName()));
   }
}

//______________________________________________________________________________
TLDAPAttribute& TLDAPAttribute::operator=(const TLDAPAttribute &attr)
{
   // Equal operator
   if(this!=&attr) {
      TNamed::operator=(attr);
      fValues=attr.fValues;
      fNCount=attr.fNCount;
   } return *this;
}

//______________________________________________________________________________
TLDAPAttribute::~TLDAPAttribute()
{
   //destructor
   delete fValues;
}

//______________________________________________________________________________
void TLDAPAttribute::AddValue(const char *value)
{
   // Add a value to the attribute.

   fValues->AddLast(new TObjString(value));
}

//______________________________________________________________________________
void TLDAPAttribute::DeleteValue(const char *value)
{
   // Delete value by name.

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

//______________________________________________________________________________
const char *TLDAPAttribute::GetValue() const
{
   // Get next value of the attribute. Returns zero after the last value,
   // then returns the first value again.

   Int_t n = GetCount();
   if (n > fNCount) {
      return ((TObjString*)fValues->At(fNCount++))->GetName();
   } else {
      fNCount = 0;
      return 0;
   }
}

//_______________________________________________________________________________
void TLDAPAttribute::Print(Option_t *) const
{
   // Print an attribute.

   Int_t counter = GetCount();
   if (counter == 0) {
      cout << GetName() << ": " << endl;
   } else if (counter != 0) {
      for (Int_t i = 0; i < counter; i++) {
         cout << GetName() << ": " << GetValue() << endl;
      }
   }
}

//_______________________________________________________________________________
LDAPMod *TLDAPAttribute::GetMod(Int_t op)
{
   // Get "LDAPMod" structure for attribute. Returned LDAPMod must be
   // deleted by the user.

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
