#ifndef __XRDOUCNAME2NAME_H__
#define __XRDOUCNAME2NAME_H__
/******************************************************************************/
/*                                                                            */
/*                    X r d O u c n a m e 2 n a m e . h h                     */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

class XrdOucName2Name
{
public:

// All of the following functions place any results in the buffer supplied by
// the first argument which is of the length supplied by the 2nd srgument. If
// the function succeeded, it must return 0. Otherwise, it must return an error
// number corresponding to one in errno.h. The most common error numbers are:

// EINVAL       - The supplied lfn (or pfn) is invalid.
// ENAMETOOLONG - The result would not fit in the supplied buffer.

// lfn2pfn() is called to map a logical file name to a physical file name.
//           If the file exists, the pfn must refer to the existing file.
//           If the file does not exist, the pfn should correspond to the name
//           of the file that could have existed. 
//
virtual int lfn2pfn(const char *lfn, char *buff, int blen) = 0;

// lfn2rfn() is called to map a logical file name to the name that the file
//           would have in a Mass Storage System (i.e., remote location). The 
//           format must be consistent with the SRM being used to access the MSS.
//           The file may or may not actually exist in the target MSS.
//
virtual int lfn2rfn(const char *lfn, char *buff, int blen) = 0;

// pfn2lfn() is called to map a physical file name to a logical file name.
//           If the file exists, the pfn refers to the existing file.
//           If the file does not exist, the pfn corresponds to the name
//           of the file that could have existed. 
//
virtual int pfn2lfn(const char *pfn, char *buff, int blen) = 0;

             XrdOucName2Name() {}
virtual     ~XrdOucName2Name() {}
};

/******************************************************************************/
/*                    X r d O u c g e t N a m e 2 N a m e                     */
/******************************************************************************/
  
// The XrdOucgetName2Name() function is called when the shared library containing
// implementation of this class is loaded. It must exist in the library as an
// 'extern "C"' defined function.

// The 1st argument is a pointer to the error object that must be used to
//         print any errors or other messages (see XrdSysError.hh).

// The 2nd argument is the name of the configuration file that was used.
//         This value may be null though that would be impossible.

// The 3rd argument is the argument string that was specified on the namelib
//         directive. It is never null but may point to a null string.

// The 4th argument is the path specified by the remoteroot directive. It is
//         a null pointer if the directive was not specified.

// The 5th argument is the path specified by the localroot directive. It is
//         a null pointer if the directive was not specified.

// The function must return an instance of this class upon success and a null
// pointer upon failure.

class XrdSysError;

#define XrdOucgetName2NameArgs XrdSysError       *eDest, \
                               const char        *confg, \
                               const char        *parms, \
                               const char        *lroot, \
                               const char        *rroot

extern "C"
{
XrdOucName2Name *XrdOucgetName2Name(XrdOucgetName2NameArgs);
}

// Warnings and admonitions!

// All object methods *must* be thread-safe!

// The Name2Name object is used frequently in the course of opening files
// as well as other meta-file operations (e.g., stat(), rename(), etc.).
// The algorithms used by this object *must* be effecient and speedy; otherwise,
// system performance will be severely degraded.
#endif
