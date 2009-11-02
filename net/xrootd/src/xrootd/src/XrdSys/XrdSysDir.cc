/******************************************************************************/
/*                                                                            */
/*                     X r d S y s D i r . h h                                */
/*                                                                            */
/* (c) 2006 G. Ganis (CERN)                                                   */
/*     All Rights Reserved. See XrdInfo.cc for complete License Terms         */
/******************************************************************************/
// $Id$

const char *XrdSysDirCVSID = "$Id$";

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdSysDir                                                            //
//                                                                      //
// Author: G. Ganis, CERN, 2006                                         //
//                                                                      //
// API for handling directories                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdSys/XrdSysDir.hh"

#if !defined(WINDOWS)
#include <dirent.h>
#else
#include <windows.h>
#endif

#include <errno.h>
#include <string.h>

//______________________________________________________________________________
XrdSysDir::XrdSysDir(const char *path)
{
   // Constructor. Initialize a directory handle for 'path'.
   // Use isValid() to check the result of this operation, and lastError()
   // to get the last error code, if any.

   lasterr = 0;
   if (path && strlen(path) > 0) {
#if !defined(WINDOWS)
      dhandle = (void *) opendir(path);
      if (!dhandle)
         lasterr = errno;
#else
      WIN32_FIND_DATA filedata;
      dhandle = (void *) ::FindFirstFile(path, &filedata);
      if ((HANDLE)dhandle == INVALID_HANDLE_VALUE) {
         lasterr = EINVAL;
         dhandle = 0;
      }
      else if (!(filedata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
         lasterr = ENOTDIR;
         dhandle = 0;
      }
#endif
   } else
      // Invalid argument
      lasterr = EINVAL;
}

//______________________________________________________________________________
XrdSysDir::~XrdSysDir()
{
   // Destructor.

   if (dhandle) {
#if !defined(WINDOWS)
      closedir((DIR *)dhandle);
#else
      ::FindClose((HANDLE)dhandle);
#endif
   }
}

//______________________________________________________________________________
char *XrdSysDir::nextEntry()
{
   // Get next entry in directory structure.
   // Return 0 if no more entries or error. In the latter case
   // the error code can be retrieved via lastError().

   char *dent = 0;

   lasterr = 0;
   if (!dhandle) {
      lasterr = EINVAL;
      return dent;
   }

#if !defined(WINDOWS)
   struct dirent *ent = readdir((DIR *)dhandle);
   if (!ent) {
      if (errno == EBADF)
         lasterr = errno;
   } else {
      dent = (char *) ent->d_name;
   }
#else
   WIN32_FIND_DATA filedata;
   if (::FindNextFile((HANDLE)dhandle, &filedata)) {
      dent = (char *) filedata.cFileName;
   } else {
      if (::GetLastError() != ERROR_NO_MORE_FILES)
         lasterr = EBADF;
   }
#endif
   // Done
   return dent;
}

