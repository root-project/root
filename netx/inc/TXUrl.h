/**********************************************************************/
/*                          T X U r l . r d l                         */
/*                                2003                                */
/*  Produced by Alvise Dorigo & Fabrizio Furano for INFN padova       */
/**********************************************************************/
//
//   $Id: TXUrl.rdl,v 1.2 2004/06/16 09:50:15 furano Exp $
//
// Author: Alvise Dorigo, Fabrizio Furano

#ifndef __TXURL_H__
#define __TXURL_H__

#include <vector>
#include "TUrl.h"
#include "TRandom.h"
#include "TString.h"

#define PROTO "root"

using namespace std;

typedef vector<TUrl*> UrlArray;

class TXUrl : public TObject {
 private:
   UrlArray fUrlArray, fTmpUrlArray;
   TString fPathName;
   
   Bool_t fIsValid;

   TRandom fRndgen;

   void CheckPort(TString &list);

   // Takes a sequence of hostname and resolves it into a vector of TUrl
   void ConvertDNSAliases(UrlArray&, TString, TString);
   Bool_t ConvertSingleDNSAlias(UrlArray&, TString, TString);

   const char *ConvertIP_to_Name(const char* IP);

 public:
   TXUrl(TString);
   ~TXUrl();

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

   ClassDef(TXUrl, 1); //A container for multiple urls.
};

#endif
