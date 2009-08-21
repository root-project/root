/******************************************************************************/
/*                                                                            */
/*                   X r d X r o o t d L o a d L i b . c c                    */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$ 

const char *XrdXrootdLoadLibCVSID = "$Id$";

// Bypass Solaris ELF madness
//
#ifdef __solaris__
#include <sys/isa_defs.h>
#if defined(_ILP32) && (_FILE_OFFSET_BITS != 32)
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#undef  _LARGEFILE_SOURCE
#endif
#endif
  
#include <dlfcn.h>
#ifndef __macos__
#include <link.h>
#endif

#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdSys/XrdSysError.hh"
  
/******************************************************************************/
/*                        G l o b a l   S y m b o l s                         */
/******************************************************************************/
  
XrdSecProtocol *(*XrdXrootdSecGetProtocol)(const char             *hostname,
                                           const struct sockaddr  &netaddr,
                                           const XrdSecParameters &parms,
                                                 XrdOucErrInfo    *einfo) = 0;

/******************************************************************************/
/*                 x r o o t d _ l o a d F i l e s y s t e m                  */
/******************************************************************************/

XrdSfsFileSystem *XrdXrootdloadFileSystem(XrdSysError *eDest, 
                                          char *fslib, const char *cfn)
{
   void *libhandle;
   XrdSfsFileSystem *(*ep)(XrdSfsFileSystem *, XrdSysLogger *, const char *);
   XrdSfsFileSystem *FS;

// Record the library path in the environment
//
   XrdOucEnv::Export("XRDOFSLIB", fslib);

// Open the file system library
//
   if (!(libhandle = dlopen(fslib, RTLD_NOW)))
      {eDest->Emsg("Config",dlerror(),"opening shared library",fslib);
       return 0;
      }

// Get the file system object creator
//
   if (!(ep = (XrdSfsFileSystem *(*)(XrdSfsFileSystem *,XrdSysLogger *,const char *))
                                  dlsym(libhandle,"XrdSfsGetFileSystem")))
      {eDest->Emsg("Config", dlerror(),
                   "finding XrdSfsGetFileSystem() in", fslib);
       return 0;
      }

// Get the file system object
//
   if (!(FS = (*ep)(0, eDest->logger(), cfn)))
      {eDest->Emsg("Config", "Unable to create file system object via",fslib);
       return 0;
      }

// All done
//
   return FS;
}
  
/******************************************************************************/
/*                   x r o o t d _ l o a d S e c u r i t y                    */
/******************************************************************************/

XrdSecService *XrdXrootdloadSecurity(XrdSysError *eDest, char *seclib, char *cfn)
{
   void *libhandle;
   XrdSecService *(*ep)(XrdSysLogger *, const char *cfn);
   XrdSecService *CIA;

// Open the security library
//
   if (!(libhandle = dlopen(seclib, RTLD_NOW)))
      {eDest->Emsg("Config",dlerror(),"opening shared library",seclib);
       return 0;
      }

// Get the server object creator
//
   if (!(ep = (XrdSecService *(*)(XrdSysLogger *, const char *cfn))dlsym(libhandle,
              "XrdSecgetService")))
      {eDest->Emsg("Config", dlerror(),
                   "finding XrdSecgetService() in", seclib);
       return 0;
      }

// Get the server object
//
   if (!(CIA = (*ep)(eDest->logger(), cfn)))
      {eDest->Emsg("Config", "Unable to create security service object via",seclib);
       return 0;
      }

// Get the client object creator (in case we are acting as a client)
//
   if (!(XrdXrootdSecGetProtocol = 
                 (XrdSecProtocol *(*)(const char             *,
                                      const struct sockaddr  &,
                                      const XrdSecParameters &,
                                            XrdOucErrInfo    *))
                 dlsym(libhandle, "XrdSecGetProtocol")))
      {eDest->Emsg("Config", dlerror(),
                   "finding XrdSecGetProtocol() in", seclib);
       return 0;
      }

// All done
//
   return CIA;
}
