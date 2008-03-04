//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientUrlInfo                                                     // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
// Alvise Dorigo, Fabrizio Furano, INFN Padova, 2003                    //
// Revised by G. Ganis, CERN,  June 2005                                //
//                                                                      //
// Class handling information about an url                              //
// The purpose of this class is to allow:                               //
//   - parsing a string url into its components                         //
//   - reading/writing the single components                            //
//   - reading the modified full url                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

#ifndef _XRC_URLINFO_H
#define _XRC_URLINFO_H

#include "XrdOuc/XrdOucString.hh"

//
// The information an url may contain
// Plus utilities for parsing and rebuilding an url
//

class XrdClientUrlInfo {
 public:
   XrdOucString Proto;
   XrdOucString Passwd;
   XrdOucString User;
   XrdOucString Host;
   int Port;
   XrdOucString HostAddr;
   XrdOucString HostWPort;
   XrdOucString File;

   void Clear();
   void TakeUrl(XrdOucString url);
   XrdOucString GetUrl();

   XrdClientUrlInfo(const char *url);
   XrdClientUrlInfo(const XrdOucString &url);
   XrdClientUrlInfo(const XrdClientUrlInfo &url);
   XrdClientUrlInfo();

   void SetAddrFromHost();

   inline bool IsValid() { return (Port >= 0); }

   XrdClientUrlInfo &operator=(const XrdOucString &url);
   XrdClientUrlInfo &operator=(const XrdClientUrlInfo &url);
};


#endif
