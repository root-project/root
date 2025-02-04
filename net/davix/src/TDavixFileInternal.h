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
   class DavFile;
}
struct Davix_fd;


class TDavixFileInternal {
   friend class TDavixFile;
   friend class TDavixSystem;

private:
   TDavixFileInternal(const TUrl & mUrl, Option_t* mopt) :
      davixContext(getDavixInstance()),
      fUrl(mUrl),
      opt(mopt),
      dirdVec() { }

   TDavixFileInternal(const char* url, Option_t* mopt) :
      davixContext(getDavixInstance()),
      fUrl(url),
      opt(mopt),
      dirdVec() { }

   ~TDavixFileInternal();

   Davix_fd *getDavixFileInstance();

   Davix_fd * Open();

   void Close();

   void enableGridMode();

   void setAwsRegion(const std::string & region);
   void setAwsToken(const std::string & token);
   void setAwsAlternate(const bool &option);

   void setS3Auth(const std::string & secret, const std::string & access,
                  const std::string & region, const std::string & token);

   void parseConfig();

   void parseParams(Option_t* option);

   void init();

   bool isMyDird(void* fd);

   void addDird(void* fd);

   void removeDird(void* fd);

   std::vector<std::string> getReplicas()
   {
     return replicas;
   }

   std::vector<std::string> replicas;

   // DAVIX
   Davix::Context *davixContext;
   Davix::RequestParams *davixParam = nullptr;
   Davix::DavPosix *davixPosix = nullptr;
   Davix_fd *davixFd = nullptr;
   TUrl fUrl;
   Option_t* opt;
   int oflags = 0;
   std::vector<void*> dirdVec;

public:
   Int_t DavixStat(const char *url, struct stat *st);

   static Davix::Context* getDavixInstance();
};

#endif
