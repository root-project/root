#ifndef __SecsssID__
#define __SecsssID__
/******************************************************************************/
/*                                                                            */
/*                        X r d S e c s s s I D . h h                         */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

#include <string.h>
#include <time.h>

#include "XrdOuc/XrdOucHash.hh"
#include "XrdSec/XrdSecEntity.hh"
#include "XrdSys/XrdSysPthread.hh"

// The XrdSecsssID class allows you to establish a registery to map loginid's
// to arbitrary entities. By default, the sss security protocol uses the
// username as the authenticated username and, if possible, the corresponding
// primary group membership of username (i.e., static mapping). The server is
// will ignore the username and/or the groupname unless the key is designated
// as anyuser, anygroup, respectively. By creating an instance of this class
// you can over-ride the default and map the loginid (i.e., the id supplied
// at login time which is normally the first 8-characters of the username or
// the id specified in the url; i.e., id@host) to arbitrary entities using
// the Register() method. You must create one, and only one, such instance
// prior to making any contact with a sss security enabled server.

// In order to include XrdSecsssID methods, you should either link with
// libXrdSecsss.so (preferable) or include XrdSecsssID.o and link with
// libXrdOuc.a and libXrdSys.a.

class XrdSecsssID
{
public:

// Register() creates a mapping from a loginid to an entity description. Only
//            name, vo, role, group, and endorements pointers in XrdSecEntity
//            are supported. To de-register a loginid, make the Ident arg zero.
//            To replace an existing entry, specify 1 for doReplace argument.
//            TRUE is returned if successful; FALSE otherwise (including the
//            case where idDynamic was not specified in the constructor or
//            doReplace is zero and the loginid has already been registered).
//
int      Register(const char *loginid, XrdSecEntity *Ident, int doReplace=0);

// Find() is an internal look-up method that returns the identification
//        string in the provided buffer corresponding to the loginid.
//        If loginid is registered and the data will fit into the buffer the
//        length moved into the buffer is returned. Otherwise, the default ID
//        is moved into the buffer and the length copied is returned. If that
//        is not possible, 0 is returned.
//
int      Find(const char *loginid, char *Buff, int Blen);

// A single instance of this class may be instantiated. The first parameter
// indicates how authentication is to be handled. The second parameter provides
// either a fixed or default authenticated identity under control of the aType
// parameter, as follows:
//
enum authType {idDynamic = 0, // Mutual: Map loginid to registered identity
                              //         Ident is default; if 0 nobody/nogroup
               idStatic  = 1, // 1Sided: fixed identity sent to the server
                              //         Ident as specified; if 0 process uid/gid
                              //         Default if XrdSecsssID not instantiated!
               idStaticM = 2  // Mutual: fixed identity sent to the server
                              //         Ident as specified; if 0 process uid/gid
              };

// getObj() returns the address of a previous created instance of this object or
//          zero if no instance exists. It also returns authType and default ID
//          to be used regardless of the return value.
//
static
XrdSecsssID *getObj(authType &aType, char **dID, int &dIDsz);

       XrdSecsssID(authType aType=idStatic, XrdSecEntity *Ident=0);

      ~XrdSecsssID() {if (defaultID) free(defaultID);}

private:

struct sssID {int iLen; char iData[1];}; // Sized appropriately
static sssID *genID(int Secure);
static sssID *genID(XrdSecEntity *eP);

static XrdSysMutex InitMutex;
       sssID      *defaultID;
XrdSysMutex        myMutex;
XrdOucHash<sssID>  Registry;
authType           myAuth;
};
#endif
