#ifndef __ACC_AUTHORIZE__
#define __ACC_AUTHORIZE__
/******************************************************************************/
/*                                                                            */
/*                    X r d A c c A u t h o r i z e . h h                     */
/*                                                                            */
/* (c) 2000 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdAcc/XrdAccPrivs.hh"

/******************************************************************************/
/*                      A c c e s s _ O p e r a t i o n                       */
/******************************************************************************/
  
// The following are supported operations
//
enum Access_Operation  {AOP_Any      = 0,  // Special for getting privs
                        AOP_Chmod    = 1,  // chmod()
                        AOP_Chown    = 2,  // chown()
                        AOP_Create   = 3,  // open() with create
                        AOP_Delete   = 4,  // rm() or rmdir()
                        AOP_Insert   = 5,  // mv() for target
                        AOP_Lock     = 6,  // n/a
                        AOP_Mkdir    = 7,  // mkdir()
                        AOP_Read     = 8,  // open() r/o, prepare()
                        AOP_Readdir  = 9,  // opendir()
                        AOP_Rename   = 10, // mv() for source
                        AOP_Stat     = 11, // exists(), stat()
                        AOP_Update   = 12, // open() r/w or append
                        AOP_LastOp   = 12  // For limits testing
                       };

/******************************************************************************/
/*                 o o a c c _ A u t h o r i z e   C l a s s                  */
/******************************************************************************/
  
class XrdOucEnv;
class XrdSecEntity;

class XrdAccAuthorize
{
public:

/* Access() indicates whether or not the user/host is permitted access to the
   path for the specified operation. The default implementation that is
   statically linked determines privileges by combining user, host, user group, 
   and user/host netgroup privileges. If the operation is AOP_Any, then the 
   actual privileges are returned and the caller may make subsequent tests using 
   Test(). Otherwise, a non-zero value is returned if access is permitted or a 
   zero value is returned is access is to be denied. Other iplementations may
   use other decision making schemes but the return values must mean the same.

   Parameters: Entity    -> Authentication information
               path      -> The logical path which is the target of oper
               oper      -> The operation being attempted (see above)
               Env       -> Environmental information at the time of the
                            operation as supplied by the path CGI string.
                            This is optional and the pointer may be zero.
*/

virtual XrdAccPrivs Access(const XrdSecEntity    *Entity,
                           const char            *path,
                           const Access_Operation oper,
                                 XrdOucEnv       *Env=0) = 0;

/* Audit() routes an audit message to the appropriate audit exit routine. See
   XrdAccAudit.h for more information on how the default implementation works.
   Currently, this method is not called by the ofs but should be used by the 
   implementation to record denials or grants, as warrented.

   Parameters: accok     -> True is access was grated; false otherwise.
               Entity    -> Authentication information
               path      -> The logical path which is the target of oper
               oper      -> The operation being attempted (see above)
               Env       -> Environmental information at the time of the
                            operation as supplied by the path CGI string.
                            This is optional and the pointer may be zero.
*/

virtual int         Audit(const int              accok,
                          const XrdSecEntity    *Entity,
                          const char            *path,
                          const Access_Operation oper,
                                XrdOucEnv       *Env=0) = 0;

// Test() check whether the specified operation is permitted. If permitted it
// returns a non-zero. Otherwise, zero is returned.
//
virtual int         Test(const XrdAccPrivs priv,
                         const Access_Operation oper) = 0;

                          XrdAccAuthorize() {}

virtual                  ~XrdAccAuthorize() {}
};
  
/******************************************************************************/
/*                   o o a c c _ A c c e s s _ O b j e c t                    */
/******************************************************************************/

class XrdSysLogger;
  
/* XrdAccAuthorizeObject() is called to obtain an instance of the auth object
   that will be used for all subsequent authorization decisions. If it returns
   a null pointer; initialization fails and the program exits. The args are:

   lp    -> XrdSysLogger to be tied to an XrdSysError object for messages
   cfn   -> The name of the configuration file
   parm  -> Parameters specified on the authlib directive. If none it is zero.
*/

extern "C" XrdAccAuthorize *XrdAccAuthorizeObject(XrdSysLogger *lp,
                                                  const char   *cfn,
                                                  const char   *parm);
#endif
