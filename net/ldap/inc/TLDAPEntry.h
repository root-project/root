// @(#)root/ldap:$Id$
// Author: Evgenia Smirnova   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLDAPEntry
#define ROOT_TLDAPEntry

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_CintLDAP
#include "CintLDAP.h"
#endif


class TLDAPServer;
class TLDAPAttribute;


class TLDAPEntry: public TObject {

friend class TLDAPServer;

private:
   TString         fDn;       // Distinguished name of entry
   TList          *fAttr;     // List of attributes
   mutable Int_t   fNCount;   // Index of attribute to be returned from GetAttribute()

   LDAPMod       **GetMods(Int_t op);  // Get array of LDAPMod structures of the entry

protected:
   TLDAPEntry& operator=(const TLDAPEntry&);

public:
   TLDAPEntry(const char *dn);
   TLDAPEntry(const TLDAPEntry &e);
   virtual ~TLDAPEntry();

   const char     *GetDn() const { return fDn; }
   void            SetDn(const char *dn) { fDn = dn; }
   void            AddAttribute(const TLDAPAttribute &attr);
   TLDAPAttribute *GetAttribute() const;
   TLDAPAttribute *GetAttribute(const char *name) const;
   void            DeleteAttribute(const char *name);
   Int_t           GetCount() const { return fAttr->GetSize(); }
   Bool_t          IsReferral() const;
   TList          *GetReferrals() const;
   void            Print(Option_t * = "") const;

   ClassDef(TLDAPEntry, 0) //describe one entry in LDAP
};

#endif
