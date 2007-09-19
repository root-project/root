// @(#)root/ldap:$Id$
// Author: Evgenia Smirnova   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLDAPAttribute
#define ROOT_TLDAPAttribute

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_CintLDAP
#include "CintLDAP.h"
#endif

class TLDAPEntry;


class TLDAPAttribute : public TNamed {

friend class TLDAPEntry;

private:
   TList          *fValues;        // list of values
   mutable Int_t   fNCount;        // next value to be returned by GetValue()

   LDAPMod    *GetMod(Int_t op);   // for getting mod for attribute

protected:
   TLDAPAttribute& operator=(const TLDAPAttribute &);

public:
   TLDAPAttribute(const char *name);
   TLDAPAttribute(const char *name, const char *value);
   TLDAPAttribute(const TLDAPAttribute &attr);
   virtual ~TLDAPAttribute();

   void            AddValue(const char *value);
   void            DeleteValue(const char *value);
   const char     *GetValue() const;
   Int_t           GetCount() const { return fValues->GetSize(); }
   void            Print(Option_t * = "") const;

   ClassDef(TLDAPAttribute, 0) //interface to LDAP
};

#endif
