// @(#)root/net:$Id$
// Author: Adrien Devresse and Tigran Mkrtchyan

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDavixFileInternal
#define ROOT_TDavixFileInternal

#include "TUrl.h"
#include "TMutex.h"

#include <vector>
#include <iterator>
#include <algorithm>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sstream>
#include <string>
#include <cstring>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDavixFileInternal                                                   //
//                                                                      //
//                                                                      //
// Support class, common to TDavixFile and TDavixSystem                 //
//                                                                      //
// Authors:     Adrien Devresse (CERN IT/SDC)                           //
//              Fabrizio Furano (CERN IT/SDC)                           //
//                                                                      //
// September 2013                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace Davix {
   class Context;
   class RequestParams;
   class DavPosix;
}
struct Davix_fd;


class TDavixFileInternal {
   friend class TDavixFile;
   friend class TDavixSystem;

private:
   TDavixFileInternal(const TUrl & mUrl, Option_t* mopt) :
      positionLock(),
      openLock(),
      davixContext(getDavixInstance()),
      davixParam(NULL),
      davixPosix(NULL),
      davixFd(NULL),
      fUrl(mUrl),
      opt(mopt),
      oflags(0),
      dirdVec() { }

   TDavixFileInternal(const char* url, Option_t* mopt) :
      positionLock(),
      openLock(),
      davixContext(getDavixInstance()),
      davixParam(NULL),
      davixPosix(NULL),
      davixFd(NULL),
      fUrl(url),
      opt(mopt),
      oflags(0),
      dirdVec() { }

   ~TDavixFileInternal();

   Davix_fd *getDavixFileInstance()
   {
      // singleton init
      if (davixFd == NULL) {
         TLockGuard l(&(openLock));
         if (davixFd == NULL) {
            davixFd = this->Open();
         }
      }
      return davixFd;
   }

   Davix_fd * Open();

   void Close();

   void enableGridMode();

   void setS3Auth(const std::string & key, const std::string & token);

   void parseConfig();

   void parseParams(Option_t* option);

   void init();

   bool isMyDird(void* fd);

   void addDird(void* fd);

   void removeDird(void* fd);

   TMutex positionLock;
   TMutex openLock;

   // DAVIX
   Davix::Context *davixContext;
   Davix::RequestParams *davixParam;
   Davix::DavPosix *davixPosix;
   Davix_fd *davixFd;
   TUrl fUrl;
   Option_t* opt;
   int oflags;
   std::vector<void*> dirdVec;

public:
   Int_t DavixStat(const char *url, struct stat *st);

   static Davix::Context* getDavixInstance();
};

#endif
