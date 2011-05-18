// @(#)root/ldap:$Id$
// Author: Oleksandr Grebenyuk   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLDAPResult
#define ROOT_TLDAPResult

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_CintLDAP
#include "CintLDAP.h"
#endif


class TLDAPServer;
class TLDAPEntry;


class TLDAPResult : public TObject {

friend class TLDAPServer;

private:
   LDAP         *fLd;              // LDAP handle of current connection
   LDAPMessage  *fSearchResult;    // Pointer to the LDAP structure that contain search results
   LDAPMessage  *fCurrentEntry;    // Pointer to the current entry to be returned from the next GetNext() call

   TLDAPEntry   *CreateEntry(LDAPMessage *entry);
   TLDAPResult() : fLd(NULL), fSearchResult(NULL), fCurrentEntry(NULL) { }
   TLDAPResult(LDAP *ld, LDAPMessage *searchresult);

protected:
   TLDAPResult(const TLDAPResult&);
   TLDAPResult& operator=(const TLDAPResult&);

public:
   virtual ~TLDAPResult();

   Int_t         GetCount() const;
   TLDAPEntry   *GetNext();
   void          Print(Option_t *option="") const;

   ClassDef(TLDAPResult, 0)  // LDAP search result set
};

#endif
