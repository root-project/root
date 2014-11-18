//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientUrlSet                                                      // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
// Alvise Dorigo, Fabrizio Furano, INFN Padova, 2003                    //
// Revised by G. Ganis, CERN, June 2005                                 //
//                                                                      //
// A container for multiple urls to be resolved through DNS aliases     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

#ifndef _XRC_URLSET_H
#define _XRC_URLSET_H

#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientVector.hh"
#include "XrdOuc/XrdOucString.hh"

using namespace std;

class XrdClientUrlInfo;

typedef XrdClientVector<XrdClientUrlInfo*> UrlArray;

//
// Manages a set of XrdClientUrlInfo objects and provides a set
// of utilities to resolve multiple addresses from the dns
// and to pick urls sequentially and randomly an url
//

class XrdClientUrlSet {
private:
   UrlArray        fUrlArray;
   UrlArray        fTmpUrlArray;
   XrdOucString    fPathName;

   bool            fIsValid;
   unsigned int    fSeed;

   void            CheckPort(int &port);
   void            ConvertDNSAlias(UrlArray& urls, XrdOucString proto,
                                   XrdOucString host, XrdOucString file);
   double          GetRandom(int seed = 0);

public:
   XrdClientUrlSet(XrdOucString urls);
   ~XrdClientUrlSet();

   // Returns the final resolved list of servers
   XrdOucString   GetServers();

   // Gets the subsequent Url, the one after the last given
   XrdClientUrlInfo *GetNextUrl();

   // From the remaining urls we pick a random one. Without reinsert.
   //  i.e. while there are not considered urls, never pick an already seen one
   XrdClientUrlInfo *GetARandomUrl();
   // Given a seed, use that to pick an url
   // the effect will be that, given the same list, the same seed will pick the same url
   XrdClientUrlInfo *GetARandomUrl(unsigned int seed);

   void              Rewind();
   void              ShowUrls();
   void              EraseUrl(XrdClientUrlInfo *url);

   // Returns the number of urls
   int               Size() { return fUrlArray.GetSize(); }

   // Returns the pathfile extracted from the CTOR's argument
   XrdOucString      GetFile() { return fPathName; }

   bool              IsValid() { return fIsValid; } // Spot malformations

};

#endif
