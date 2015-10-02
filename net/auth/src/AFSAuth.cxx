// @(#)root/auth:$Id$
// Author: G. Ganis, Nov 2006

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AFSAuth                                                              //
//                                                                      //
// Utility functions to acquire and handle AFS tokens.                  //
// These functions are available as separate plugin, libAFSAuth.so,     //
// depending aonly on the AFS libraries.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string.h>

#include "AFSAuth.h"

extern "C" {
#include <afs/stds.h>
#include <afs/kautils.h>
#include <afs/com_err.h>
afs_int32 ka_Authenticate(char *name, char *instance, char *cell,
                          struct ubik_client *conn, int service,
                          struct ktc_encryptionKey *key, Date start,
                          Date end, struct ktc_token *token,
                          afs_int32 * pwexpires);
afs_int32 ka_AuthServerConn(char *cell, int service,
                            struct ktc_token *token,
                            struct ubik_client **conn);
afs_int32 ka_GetAuthToken(char *name, char *instance, char *cell,
                          struct ktc_encryptionKey *key,
                          afs_int32 lifetime, afs_int32 *pwexpires);
afs_int32 ka_GetAFSTicket(char *name, char *instance, char *realm,
                          Date lifetime, afs_int32 flags);
char *ka_LocalCell();
void      ka_StringToKey(char *str, char *cell,
                         struct ktc_encryptionKey *key);
int ktc_GetToken(struct ktc_principal *server, struct ktc_token *token,
                 int tokenLen, struct ktc_principal *client);

typedef struct ktc_token AFStoken_t;

//________________________________________________________________________
char *GetAFSErrorString(afs_int32 rc)
{
   // Decode the error code returning a pointer to a human
   // readable string. The two additional messages are taken from the OpenAFS
   // source (src/kauth/user.c).

   const char *emsg = 0;
   if (rc) {
      switch (rc) {
         case KABADREQUEST:
            emsg = "password was incorrect";
            break;
         case KAUBIKCALL:
            emsg = "Authentication Server was unavailable";
            break;
         default:
#ifdef R__AFSOLDCOMERR
            emsg = error_message(rc);
#else
            emsg = afs_error_message(rc);
#endif
      }
   } else {
      emsg = "";
   }

   // Done
   return (char *)emsg;
}


//________________________________________________________________________
void *GetAFSToken(const char *usr, const char *pwd, int pwlen,
                  int life, char **emsg)
{
   // Get AFS token for the local cell for 'usr'. The meaning of the
   // information passed at 'pwd' depends on 'pwlen'. For 'pwlen <= 0'
   // 'pwd' is interpreted as the plain password (null terminated string).
   // For 'pwlen > 0', the 'pwlen' bytes at 'pwd' contain the password in
   // for of encryption key (struct ktc_encryptionKey).
   // On success a token is returned as opaque information.
   // On error / failure, 0 is returned; if emsg != 0, *emsg points to an
   // error message.

   // reset the error message, if defined
   if (emsg)
      *emsg = (char *)"";

   // Check user name
   if (!usr || strlen(usr) <= 0) {
      if (emsg)
         *emsg = (char *)"Input user name undefined - check your inputs!";
      return (void *)0;
   }

   // Check password buffer
   if (!pwd || (pwlen <= 0 && strlen(pwd) <= 0)) {
      if (emsg)
         *emsg = (char *)"Password buffer undefined - check your inputs!";
      return (void *)0;
   }

   // Check lifetime
   if (life < 0) {
      // Use default
      life = DFLTTOKENLIFETIME;
   } else if (life == 0) {
      // Shortest possible (5 min; smaller values are ignored)
      life = 300;
   }

   // Init error tables and connect to the correct CellServDB file
   afs_int32 rc = 0;
   if ((rc = ka_Init(0))) {
      // Failure
      if (emsg)
         *emsg = (char *)GetAFSErrorString(rc);
      return (void *)0;
   }

   // Fill the encryption key
   struct ktc_encryptionKey key;
   if (pwlen > 0) {
      // Just copy the input buffer
      memcpy(key.data, pwd, pwlen);
   } else {
      // Get rid of '\n', if any
      int len = strlen(pwd);
      if (pwd[len-1] == '\n')
         len--;
      char *pw = new char[len + 1];
      memcpy(pw, pwd, len);
      pw[len] = 0;
      // Create the key from the password
      ka_StringToKey(pw, 0, &key);
      delete[] pw;
   }

   // Get the cell
   char *cell = 0;
   char cellname[MAXKTCREALMLEN];
   if (ka_ExpandCell(cell, cellname, 0) != 0) {
      if (emsg)
         *emsg = (char *)"Could not expand cell name";
      return (void *)0;
   }
   cell = cellname;

   // Get an unauthenticated connection to desired cell
   struct ubik_client *conn = 0;
   if (ka_AuthServerConn(cell, KA_AUTHENTICATION_SERVICE, 0, &conn) != 0) {
      if (emsg)
         *emsg = (char *)"Could not get a connection to server";
      return (void *)0;
   }

   // Authenticate now
   AFStoken_t *tkn = new AFStoken_t;
   int pwexpires;
   int now = time(0);
   rc = 0;
   if ((rc = ka_Authenticate((char *)usr, (char *)"", cell, conn,
                             KA_TICKET_GRANTING_SERVICE,
                             &key, now, now + life, tkn, &pwexpires))) {
      // Failure
      if (emsg)
         *emsg = (char *)GetAFSErrorString(rc);
      ubik_ClientDestroy(conn);
      return (void *)0;
   }

   // Now get a ticket to access user's AFS private area
   if ((rc = ka_GetAuthToken((char *)usr, "", "", &key, life, &pwexpires))) {
      // Failure
      if (emsg)
         *emsg = (char *)GetAFSErrorString(rc);
      ubik_ClientDestroy(conn);
      return (void *)0;
   }
   if ((rc = ka_GetAFSTicket((char *)usr, "", "", life,
                             KA_USERAUTH_VERSION + KA_USERAUTH_DOSETPAG))) {
      // Failure
      if (emsg)
         *emsg = (char *)GetAFSErrorString(rc);
      ubik_ClientDestroy(conn);
      return (void *)0;
   }

   // Release the connection to the server
   ubik_ClientDestroy(conn);

   // Success
   return (void *)tkn;
}

//________________________________________________________________________
int VerifyAFSToken(void *token)
{
   // Verify validity an AFS token. The opaque input information is the one
   // returned by a successful call to GetAFSToken.
   // The remaining lifetime is returned, i.e. <=0 if expired.

   // Check input
   if (!token)
      return 0;

   // Unveil it
   AFStoken_t *tkn = (AFStoken_t *) token;

   // Compare expiration time with now
   return ((int) tkn->endTime - time(0));

}

//________________________________________________________________________
void DeleteAFSToken(void *token)
{
   // Delete an AFS token returned by a successful call to GetAFSToken.

   if (token)
      delete (AFStoken_t *)token;
}

//________________________________________________________________________
char *AFSLocalCell()
{
   // Returns a pointer to a string with the local cell. The string must
   // not be freed or deleted.

   return ka_LocalCell();
}

}
