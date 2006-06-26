// @(#)root/monalisa:$Name:$:$Id:$
// Author: Andreas Peters   5/10/2005

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
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

#include "TMonaLisaReader.h"
#include "TObjString.h"
#include "TMap.h"
#include "Riostream.h"


ClassImp(TMonaLisaReader)

//______________________________________________________________________________
TMonaLisaReader::TMonaLisaReader(const char* serviceurl) :
   MonaLisaWsClient(serviceurl), fWebservice(serviceurl)

{
   // Creates a TMonaLisaReader object to read monitoring information from a
   // from a MonaLisa server using the ML web service client package
   // (libmonalisawsclient.so / SOAP client).
   //
   // To use TMonaLisaReader, libMonaLisa.so has to be loaded.

   gMonitoringReader = this;
}

//______________________________________________________________________________
TMonaLisaReader::~TMonaLisaReader()
{
   // Cleanup.

}

//______________________________________________________________________________
void TMonaLisaReader::Print(Option_t *) const
{
   // Print info about MonaLisa object.

   cout << "Webservice URL: " << serviceUrl << endl;
}

//______________________________________________________________________________
TMap *TMonaLisaReader::GetMap()
{
   // Return the MonaLisa result in a ROOT map.

   Tresult::iterator riter;

   TMap *rmap = new TMap();
   for (riter=result.begin(); riter!=result.end(); ++riter) {
      Tfarm::iterator fiter;
      TMap *fmap = new TMap();
      for (fiter= (riter->second).begin(); fiter!=(riter->second).end(); ++fiter) {
         Tcluster::iterator citer;
         TMap *cmap = new TMap();
         for (citer= (fiter->second).begin(); citer!=(fiter->second).end(); ++citer) {
            Tnode::iterator niter;
            TMap* nmap = new TMap();
            for (niter= (citer->second).begin(); niter!=(citer->second).end(); ++niter) {
               Tparameter::iterator piter;
               TMap *pmap = new TMap();
               for (piter= (niter->second).begin(); piter!=(niter->second).end(); ++piter) {
                  char valtimestr[1024];
                  sprintf(valtimestr,"%ld\n",piter->first);
                  pmap->Add(new TObjString(valtimestr), new TObjString(piter->second.c_str()));
               }
               nmap->Add(new TObjString(niter->first.c_str()),pmap);
            }
            cmap->Add(new TObjString(citer->first.c_str()),nmap);
         }
         fmap->Add(new TObjString(fiter->first.c_str()),cmap);
      }
      rmap->Add(new TObjString(riter->first.c_str()),fmap);
   }
   return rmap;
}

//______________________________________________________________________________
void  TMonaLisaReader::DeleteMap(TMap *map)
{
   // Delete the map.

   if (map) {
      map->DeleteAll();
      delete map;
   }
}

//______________________________________________________________________________
void TMonaLisaReader::ProxyValues(const char *farmName, const char *clusterName,
                                  const char *nodeName, const char *paramName,
                                  Long_t min, Long_t max, Long_t lifetime)
{
   // Execute GetValues() and accumulate/proxy the results for the
   // lifetime given as lifetime.

   Long_t newesttime=0;

   // Copy the previous state
   Tresult oldresult;
   oldresult.swap(result);

   // Get the new values
   GetValues(farmName,clusterName,nodeName,paramName,min,max);
   // Add the new state to the old state

   Tresult::iterator riter;
   for (riter=result.begin(); riter!=result.end(); ++riter) {
      Tfarm::iterator fiter;
      for (fiter= (riter->second).begin(); fiter!=(riter->second).end(); ++fiter) {
         Tcluster::iterator citer;
         for (citer= (fiter->second).begin(); citer!=(fiter->second).end(); ++citer) {
            Tnode::iterator niter;
            for (niter= (citer->second).begin(); niter!=(citer->second).end(); ++niter) {
               //std::cout << riter->first << "::" << fiter->first << "::" << citer->first << "::" << niter->first << std::endl;
               Tparameter::iterator piter;
               for (piter= (niter->second).begin(); piter!=(niter->second).end(); ++piter) {
                  if (piter->first > newesttime) {
                     newesttime = piter->first;
                  }
                  if (oldresult[riter->first][fiter->first][citer->first][niter->first][piter->first] != "") {
                     //std::cout << "Value Exists  " << piter->first << " : " << piter->second << std::endl;
                  } else {
                     //std::cout << "Adding New    " << piter->first << " : " << piter->second << std::endl;
                     oldresult[riter->first][fiter->first][citer->first][niter->first][piter->first]=piter->second;
                  }
               }
            }
         }
      }
   }

   // loop to eliminate expired info's
   for (riter=oldresult.begin(); riter!=oldresult.end(); ++riter) {
      Tfarm::iterator fiter;
      for (fiter= (riter->second).begin(); fiter!=(riter->second).end(); ++fiter) {
         Tcluster::iterator citer;
         for (citer= (fiter->second).begin(); citer!=(fiter->second).end(); ++citer) {
            Tnode::iterator niter;
            for (niter= (citer->second).begin(); niter!=(citer->second).end(); ++niter) {
               Tparameter::iterator piter;
               for (piter= (niter->second).begin(); piter!=(niter->second).end(); ++piter) {
                  if (piter->first < (newesttime-lifetime)) {
                     std::cout << "Eliminating entry " << piter->first << " : " << piter->second << std::endl;
                     oldresult[riter->first][fiter->first][citer->first][niter->first].erase(piter->first);
                  }
               }
            }
         }
      }
   }

   // copy back the filled result to the result structure
   result.swap(oldresult);
}
