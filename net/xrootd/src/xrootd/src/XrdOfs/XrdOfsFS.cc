/******************************************************************************/
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdOfsFSCVSID = "$Id$";
  
// If you are replacing the standard definition of the file system interface,
// with a derived class to perform additional or enhanced functions, you MUST
// define XrdOfsFS to be an instance of your derived class definition. You
// would then create a shared library linking against libXrdOfs.a and manually
// include your definition of XrdOfsFS (obviously upcast to XrdOfs). This
// is how the standard libXrdOfs.so is built.

// If additional configuration is needed, over-ride the Config() method. At the
// the end of your config, return the result of the XrdOfs::Config().

#include "XrdOfs/XrdOfs.hh"
#include "XrdSys/XrdSysPthread.hh"

XrdOfs XrdOfsFS;
