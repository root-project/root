// @(#)root/monalisa:$Name:$:$Id:$
// Author: Andreas Peters   5/10/2005

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMonaLisaReader
#define ROOT_TMonaLisaReader

#ifndef ROOT_TVirtualMonitoring
#include "TVirtualMonitoring.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif


#ifndef __CINT__
#include <monalisawsclient.h>
#else
class MonaLisaWsClient;
#endif

class TMap;


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMonaLisaReader                                                      //
//                                                                      //
// Class defining interface to MonaLisa Monitoring Services in ROOT.    //
// The TMonaLisaReader object is used to read monitoring information    //
// from a MonaLisa server using the ML web service client package       //
// (libmonalisawsclient.so / SOAP client).                              //
// The MonaLisa web service client library for C++ can be downloaded at //
// http://monalisa.cacr.caltech.edu/monalisa__Download__wsclient.html,  //
// current version:                                                     //
// http://monalisa.cacr.caltech.edu/download/monalisa/ \                //
//        ml-gsoapclient-1.0.0.tar.gz                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMonaLisaReader : public TVirtualMonitoringReader, MonaLisaWsClient {

public:
   TUrl fWebservice;  // address of MonaLisa web service

   TMonaLisaReader(const char* serviceurl="");
   virtual ~TMonaLisaReader();

   void   Print(Option_t *option = "") const;

   void   DumpResult() { MonaLisaWsClient::Dump(); }
   void   GetValues(const char *farmName, const char *clusterName,
                    const char *nodeName, const char *paramName,
                    Long_t min, Long_t max, Bool_t debug=kFALSE) {
      getValues(farmName,clusterName,nodeName,paramName,min, max, debug); }
   void GetLastValues(const char *farmName, const char *clusterName,
                      const char *nodeName, const char *paramName,
                      Bool_t debug=kFALSE) {
      getFilteredLastValues(farmName, clusterName, nodeName, paramName, debug); }
   void ProxyValues(const char *farmName, const char *clusterName,
                    const char *nodeName, const char *paramName,
                    Long_t min, Long_t max, Long_t lifetime);

   TMap *GetMap();
   void  DeleteMap(TMap *map);

   ClassDef(TMonaLisaReader, 1)   // Interface to MonaLisa Monitoring
};

#endif
