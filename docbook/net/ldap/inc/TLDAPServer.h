// @(#)root/ldap:$Id$
// Author: Oleksandr Grebenyuk   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLDAPServer
#define ROOT_TLDAPServer

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_CintLDAP
#include "CintLDAP.h"
#endif

class TList;
class TLDAPResult;
class TLDAPEntry;


class TLDAPServer : public TObject {

private:
   LDAP         *fLd;           // LDAP handle of current connection
   TString       fBinddn;       // Bind name
   TString       fPassword;     // Password
   Bool_t        fIsConnected;  // Current connection state

   Int_t         Bind();
   void          Unbind();

   static void   DeleteMods(LDAPMod **mods);

protected:
   TLDAPServer(const TLDAPServer&);
   TLDAPServer& operator=(const TLDAPServer&);

public:
   TLDAPServer(const char *host, Int_t port = LDAP_PORT,
               const char *binddn = 0, const char *password = 0,
               Int_t version = LDAP_VERSION2);

   virtual ~TLDAPServer();

   Bool_t        IsConnected() const { return fIsConnected; };
   TLDAPResult  *Search(const char *base = "",
                        Int_t scope = LDAP_SCOPE_BASE,
                        const char *filter = 0,
                        TList *attrs = 0,
                        Bool_t attrsonly = 0);
   const char   *GetNamingContexts();
   const char   *GetSubschemaSubentry();
   TLDAPResult  *GetObjectClasses();
   TLDAPResult  *GetAttributeTypes();

   Int_t         AddEntry(TLDAPEntry &entry);
   Int_t         ModifyEntry(TLDAPEntry &entry, Int_t mode = LDAP_MOD_REPLACE);
   Int_t         DeleteEntry(const char *dn);
   Int_t         RenameEntry(const char *dn, const char *newrdn,
                             Bool_t removeattr = kFALSE);

   ClassDef(TLDAPServer, 0)  // Connection to LDAP server
};

#endif
