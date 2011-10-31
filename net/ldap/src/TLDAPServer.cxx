// @(#)root/ldap:$Id$
// Author: Oleksandr Grebenyuk   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLDAPServer.h"
#include "TLDAPResult.h"
#include "TLDAPEntry.h"
#include "TLDAPAttribute.h"
#include "TObjString.h"
#include "TList.h"
#include "TError.h"


ClassImp(TLDAPServer)

//______________________________________________________________________________
TLDAPServer::TLDAPServer(const char *host, Int_t port, const char *binddn,
                         const char *password, Int_t version)
{
   // During construction TLDAPServer object tries to connect to the
   // specified server and you should check the connection status by
   // calling the IsConnected() member function immediately after
   // creating that object.
   // const char *host:     The name of host to connect. Default is "localhost".
   // Int_t port:           Port number to connect. Default is LDAP_PORT (=389).
   // const char *binddn:   Bind DN.
   // const char *password: Password. Usually you have to specify bind DN and
   //                       password to have the write permissions. Default
   //                       values for bind DN and password are zero, that means
   //                       anonymous connection. Usually it is enough to read
   //                       the data from the server.
   //  Int_t version        Set LDAP protocol version: LDAP_VERSION1,
   //                       LDAP_VERSION2, LDAP_VERSION3

   fLd          = 0;
   fIsConnected = kFALSE;
   fBinddn      = binddn;
   fPassword    = password;

   fLd = ldap_init(host, port);
   if (!fLd) {
      Error("TLDAPServer", "error in ldap_init function");
   } else {
      if (ldap_set_option(fLd, LDAP_OPT_PROTOCOL_VERSION, &version) != LDAP_OPT_SUCCESS ) {
         Error("Bind", "Could not set protocol version!");
         return;
      }

      Bind( );
   }
}

//______________________________________________________________________________
TLDAPServer::TLDAPServer(const TLDAPServer& lds) :
   TObject(lds),
   fLd(lds.fLd),
   fBinddn(lds.fBinddn),
   fPassword(lds.fPassword),
   fIsConnected(lds.fIsConnected)
{
   // Copy constructor
}

//______________________________________________________________________________
TLDAPServer& TLDAPServer::operator=(const TLDAPServer& lds)
{
   // Equal operator
   if(this!=&lds) {
      TObject::operator=(lds);
      fLd=lds.fLd;
      fBinddn=lds.fBinddn;
      fPassword=lds.fPassword;
      fIsConnected=lds.fIsConnected;
   } return *this;
}

//______________________________________________________________________________
TLDAPServer::~TLDAPServer()
{
   // If the object is connected to the server, it disconnects.

   Unbind();
}

//______________________________________________________________________________
Int_t TLDAPServer::Bind()
{
   // Binds to the server with specified binddn and password.
   // Return value: LDAP error code, 0 if successfully bound.

   if (!IsConnected()) {
      Int_t result = ldap_simple_bind_s(fLd, fBinddn.Data(), fPassword.Data());
      if (result != LDAP_SUCCESS) {
         ldap_unbind(fLd);
         fIsConnected = kFALSE;
         switch (result) {
            case LDAP_INVALID_CREDENTIALS:
               Error("Bind", "invalid password");
               break;
            case LDAP_INAPPROPRIATE_AUTH:
               Error("Bind", "entry has no password to check");
               break;
            default :
               Error("Bind", "%s", ldap_err2string(result));
               break;
         }
      } else {
         fIsConnected = kTRUE;
      }
      return result;
   }
   return 0;
}

//______________________________________________________________________________
void TLDAPServer::Unbind()
{
   // Unbinds from the server with specified binddn and password.

   if (IsConnected()) {
      ldap_unbind(fLd);
      fIsConnected = kFALSE;
   }
}

//______________________________________________________________________________
const char *TLDAPServer::GetNamingContexts()
{
   // Performs an LDAPSearch with the attribute "namingContexts" to be
   // returned with the result. The value of this attribute is
   // extracted and returned as const char.

   TList *attrs = new TList;
   attrs->SetOwner();
   attrs->AddLast(new TObjString("namingContexts"));
   const char *namingcontexts = 0;

   TLDAPResult *result = Search("", LDAP_SCOPE_BASE, 0, attrs, 0);

   if (result) {
      TLDAPEntry *entry = result->GetNext();

      TLDAPAttribute *attribute = entry->GetAttribute();

      if (attribute)
         namingcontexts = attribute->GetValue();

      delete entry;
      delete result;
   }
   delete attrs;

   return namingcontexts;
}

//______________________________________________________________________________
const char *TLDAPServer::GetSubschemaSubentry()
{
   // Performs an LDAPSearch with the attribute "subschemaSubentry" to
   // be returned with the result. The value of this attribute is
   // extracted and returned as const char.

   TList *attrs = new TList;
   attrs->SetOwner();
   attrs->AddLast(new TObjString("subschemaSubentry"));
   const char *subschema = 0;

   TLDAPResult *result = Search("", LDAP_SCOPE_BASE, 0, attrs, 0);

   if (result) {
      TLDAPEntry *entry = result->GetNext();

      TLDAPAttribute *attribute = entry->GetAttribute();
      if (attribute)
         subschema = attribute->GetValue();

      delete entry;
      delete result;
   }
   delete attrs;

   return subschema;
}

//______________________________________________________________________________
TLDAPResult *TLDAPServer::GetObjectClasses()
{
   // Calls GetSubschemaSubentry() and performs and LDAPSearch with
   // the attribute "objectClasses" to be returned with the result.
   // The returned result object must be deleted by the user.

   const char *subschema = GetSubschemaSubentry();

   TList *attrs = new TList;
   attrs->SetOwner();
   attrs->AddLast(new TObjString("objectClasses"));

   TLDAPResult *result = Search(subschema, LDAP_SCOPE_BASE, 0, attrs, 0);

   delete attrs;

   return result;
}

//______________________________________________________________________________
TLDAPResult *TLDAPServer::GetAttributeTypes()
{
   // Calls GetSubschemaSubentry() and performs and LDAPSearch with the
   // attribute "attributeTypes" to be returned with the result.
   // The returned result object must be deleted by the user.

   const char *subschema = GetSubschemaSubentry();

   TList *attrs = new TList;
   attrs->SetOwner();
   attrs->AddLast(new TObjString("attributeTypes"));

   TLDAPResult *result = Search(subschema, LDAP_SCOPE_BASE, 0, attrs, 0);

   delete attrs;

   return result;
}

//______________________________________________________________________________
TLDAPResult *TLDAPServer::Search(const char *base, Int_t scope,
                                 const char *filter, TList *attrs,
                                 Bool_t attrsonly)
{
   // Performs searching at the LDAP directory.
   // Return value:     a TLDAPResult object or 0 in case of error.
   //                   Result needs to be deleted by user.
   // const char *base: Specifies the base object for the search operation
   // Int_t scope:      Specifies the portion of the LDAP tree, relative to
   //                   the base object, to search.
   //                   Must be one of LDAP_SCOPE_BASE (==0),
   //                   LDAP_SCOPE_ONELEVEL (==1) or LDAP_SCOPE_SUBTREE (==2).
   // char *filter:     The criteria during the search to determine which
   //                   entries to return, 0 means that the filter
   //                   "(objectclass=*)" will be applied
   // TList *attrs:     The TList of attributes to be returned along with
   //                   each entry, 0 means that all available attributes
   //                   should be returned.
   // Int_t attrsonly:  This parameter is a boolean specifying whether both
   //                   types and values should be returned with each
   //                   attribute (zero) or types only should be returned
   //                   (non-zero).

   Bind();

   Int_t errcode;
   TLDAPResult *result = 0;

   if (IsConnected()) {

      LDAPMessage *searchresult;
      char **attrslist = 0;
      if (attrs) {
         Int_t n = attrs->GetSize();
         attrslist = new char* [n + 1];
         for (Int_t i = 0; i < n; i++)
            attrslist[i] = (char*) ((TObjString*)attrs->At(i))->GetName();
         attrslist[n] = 0;
      }
      if (filter == 0)
         filter = "(objectClass=*)";

      errcode = ldap_search_s(fLd, base, scope, filter, attrslist,
                              attrsonly, &searchresult);

      delete [] attrslist;

      if (errcode == LDAP_SUCCESS) {
         result = new TLDAPResult(fLd, searchresult);
      } else {
         ldap_msgfree(searchresult);
         Error("Search", "%s", ldap_err2string(errcode));
      }

   } else {
      errcode = LDAP_SERVER_DOWN;
      Error("Search", "%s", "server is not connected");
   }

   return result;
}

//______________________________________________________________________________
Int_t TLDAPServer::AddEntry(TLDAPEntry &entry)
{
   // Adds entry to the LDAP tree.
   // Be sure that you are bound with write permissions.
   // Return value: LDAP error code.

   Bind();

   Int_t errcode;
   if (IsConnected()) {
      LDAPMod **ms = entry.GetMods(0);
      errcode = ldap_add_s(fLd, entry.GetDn(), ms);
      TLDAPServer::DeleteMods(ms);
      if (errcode != LDAP_SUCCESS)
         Error("AddEntry", "%s", ldap_err2string(errcode));
   } else {
      errcode = LDAP_SERVER_DOWN;
      Error("AddEntry", "server is not connected");
   }
   return errcode;
}

//______________________________________________________________________________
Int_t TLDAPServer::ModifyEntry(TLDAPEntry &entry, Int_t mode)
{
   // Modifies specified entry.
   // Be sure that you are bound with write permissions.
   // Return value:      LDAP error code, 0 = success.
   // TLDAPEntry &entry: Entry to be modified.
   // Int_t mode:        Modifying mode.
   //                    Should be one of LDAP_MOD_ADD (==0),
   //                    LDAP_MOD_DELETE (==1) or LDAP_MOD_REPLACE (==2)
   //                    Specifies what to do with all the entry's attributes
   //                    and its values - add to the corresponding entry on
   //                    the server, delete from it, or replace the
   //                    corresponding attributes with new values

   Bind();

   Int_t errcode;
   if (IsConnected()) {
      LDAPMod **ms = entry.GetMods(mode);
      errcode = ldap_modify_s(fLd, entry.GetDn(), ms);
      TLDAPServer::DeleteMods(ms);
      if (errcode != LDAP_SUCCESS)
         Error("ModifyEntry", "%s", ldap_err2string(errcode));
   } else {
      errcode = LDAP_SERVER_DOWN;
      Error("ModifyEntry", "server is not connected");
   }
   return errcode;
}

//______________________________________________________________________________
Int_t TLDAPServer::DeleteEntry(const char *dn)
{
   // Deletes the entry with specified DN, the base entry must exist.
   // Be sure that you are bound with write permissions.
   // Return value: LDAP error code, 0 = succes.

   Bind();

   Int_t errcode;
   if (IsConnected()) {
      errcode = ldap_delete_s(fLd, dn);
      if (errcode != LDAP_SUCCESS)
         Error("DeleteEntry", "%s", ldap_err2string(errcode));
   } else {
      errcode = LDAP_SERVER_DOWN;
      Error("DeleteEntry", "server is not connected");
   }
   return errcode;
}

//______________________________________________________________________________
Int_t TLDAPServer::RenameEntry(const char *dn, const char *newrdn, Bool_t removeattr)
{
   // Renames the entry with specified DN, the entry must be leaf
   // Be sure that you are bound with the write permissions
   // Return value:      LDAP error code, 0 = succes
   // char *dn:          Distinguished name of entry to be renamed.
   //                    This entry must be a leaf in the LDAP directory tree.
   // char *newrdn:      The new relative distinguished name to give the entry
   //                    being renamed.
   // Bool_t removeattr: This parameter specifies whether or not the
   //                    attribute values in the old relative distinguished
   //                    name should be removed from the entry
   //                    or retained as non-distinguished attributes.

   Int_t errcode;
   if (IsConnected()) {
      errcode = ldap_modrdn2_s(fLd, dn, newrdn, removeattr);
      if (errcode != LDAP_SUCCESS)
         Error( "RenameEntry", "%s", ldap_err2string(errcode));
   } else {
      errcode = LDAP_SERVER_DOWN;
      Error("RenameEntry", "server is not connected");
   }
   return errcode;
}

//______________________________________________________________________________
void TLDAPServer::DeleteMods(LDAPMod **mods)
{
   // Deletes the array of LDAPMod structures and frees its memory.
   // LDAPMod **mods: Pointer to the zero-terminated array of pointers
   //                 to LDAPMod structures

#if 1
   ldap_mods_free(mods, 1);
#else
   Int_t i = 0;
   LDAPMod *mod;
   while ((mod = mods[i++]) != 0) {
      if (mod->mod_op & LDAP_MOD_BVALUES) {
         ber_bvecfree(mod->mod_bvalues);
      } else {
         Int_t j = 0;
         char *c;
         while ((c = mod->mod_values[j++]) != 0)
            delete c;
      }
      delete mod->mod_type;
      delete mod;
   }
   delete mods;
#endif
}
