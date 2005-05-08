// @(#)root/netx:$Name:  $:$Id: TXUrl.h,v 1.4 2004/12/08 14:34:18 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXUrl
#define ROOT_TXUrl

#include <vector>

#ifndef ROOT_TUrl
#include "TUrl.h"
#endif
#ifndef ROOT_TRandom
#include "TRandom.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

#define PROTO "root"


typedef std::vector<TUrl*> UrlArray;


class TXUrl {

private:
   UrlArray fUrlArray, fTmpUrlArray;
   TString fPathName;

   Bool_t fIsValid;

   TRandom fRndgen;

   void CheckPort(TString &list);

   // Takes a sequence of hostname and resolves it into a vector of TUrl
   void ConvertDNSAliases(UrlArray&, TString, TString);
   Bool_t ConvertSingleDNSAlias(UrlArray&, TString, TString);

 public:
   TXUrl(TString);
   virtual ~TXUrl();

   // Returns the final resolved list of servers
   TString GetServers() {
      TString s;
      for ( Int_t i = 0; i < (Int_t)fUrlArray.size(); i++ )
	 s += fUrlArray[i]->GetHost();
      return s;
   }

   // Gets the subsequent Url, the one after the last given
   TUrl *GetNextUrl();

   // From the remaining urls we pick a random one. Without reinsert.
   //  i.e. while there are not considered urls, never pick an already seen one
   TUrl *GetARandomUrl();

   void Rewind(void);
   void ShowUrls(void);

   // Returns the number of urls
   Int_t Size(void) { return fUrlArray.size(); }

   // Returns the pathfile extracted from the CTOR's argument
   const char *GetFile(void) { return fPathName.Data(); }

   Bool_t IsValid(void) { return fIsValid; }    // Return kFALSE if the CTOR's argument is malformed

   ClassDef(TXUrl, 0); //A container for multiple urls.
};

#endif
