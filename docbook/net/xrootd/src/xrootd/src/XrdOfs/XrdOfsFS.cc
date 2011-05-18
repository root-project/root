/******************************************************************************/
/*                                                                            */
/*            X r d S f s G e t D e f a u l t F i l e S y s t e m             */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOfs/XrdOfs.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"

// If you are replacing the standard definition of the file system interface,
// with a derived class to perform additional or enhanced functions, you MUST
// define XrdOfsFS to be an instance of your derived class definition. You
// would then create a shared library linking against libXrdOfs.a and manually
// include your definition of XrdOfsFS (obviously upcast to XrdOfs). This
// is how the standard libXrdOfs.so is built.

// If additional configuration is needed, over-ride the Config() method. At the
// the end of your config, return the result of the XrdOfs::Config().

XrdOfs XrdOfsFS;
  
XrdSfsFileSystem *XrdSfsGetDefaultFileSystem(XrdSfsFileSystem *native_fs,
                                             XrdSysLogger     *lp,
                                             const char       *configfn)
{
   extern XrdSysError OfsEroute;

// No need to herald this as it's now the default filesystem
//
   OfsEroute.SetPrefix("ofs_");
   OfsEroute.logger(lp);

// Initialize the subsystems
//
   XrdOfsFS.ConfigFN = (configfn && *configfn ? strdup(configfn) : 0);
   if ( XrdOfsFS.Configure(OfsEroute) ) return 0;

// All done, we can return the callout vector to these routines.
//
   return &XrdOfsFS;
}
