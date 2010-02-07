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

//         $Id$

const char *XrdClientUrlSetCVSID = "$Id$";

#include <XrdClient/XrdClientUrlSet.hh>
#include "XrdClient/XrdClientUrlInfo.hh"
#include <XrdNet/XrdNetDNS.hh>
#include "XrdSys/XrdSysHeaders.hh"

#include <math.h>
#include <stdio.h>
#include <ctype.h>               // needed by isdigit()
#ifndef WIN32
#include <netdb.h>               // needed by getservbyname()
#include <netinet/in.h>          // needed by ntohs()

#include <stdlib.h>
#include <resolv.h>
#include <sys/time.h>
#include <unistd.h>
#else
#include <stdlib.h>
#include "XrdSys/XrdWin32.hh"
#include <process.h>
#endif

#include "XrdClient/XrdClientDebug.hh"

#ifdef __solaris__
#include <sunmath.h>
#endif


using namespace std;


//_____________________________________________________________________________
XrdOucString XrdClientUrlSet::GetServers()
{
   // Returns the final resolved list of servers
   XrdOucString s;

   for ( int i = 0; i < fUrlArray.GetSize(); i++ ) {
      s += fUrlArray[i]->Host;
      s += "\n";
   }

   return s;
}

//_____________________________________________________________________________
double XrdClientUrlSet::GetRandom(int i)
{
//  Machine independent random number generator.
//  Produces uniformly-distributed floating points between 0 and 1.
//  Identical sequence on all machines of >= 32 bits.
//  Periodicity = 10**8
//  Universal version (Fred James 1985).
//  generates a number in ]0,1]

   const double kCONS = 4.6566128730774E-10;
   const int kMASK24  = 2147483392;

   fSeed *= 69069;      
   unsigned int jy = (fSeed&kMASK24); // Set lower 8 bits to zero to assure exact float
   if (jy) return kCONS*jy;
   return GetRandom();
}

//_____________________________________________________________________________
XrdClientUrlSet::XrdClientUrlSet(XrdOucString urls) : fIsValid(TRUE)
{
   // A container for multiple urls.
   // It creates an array of multiple urls parsing the argument 'urls' and
   // resolving the DNS aliases
   //
   // 'urls' MUST be in the form:
   //
   //             [proto://][user1@]host1:port1[,[user2@]host2:port2, ... ,
   //                       [userN@]hostN:portN]]/pathfile
   //
   // Using the method GetNextUrl() the user can obtain the next
   // XrdClientUrlInfo object pointer in the array (the array is cyclic).
   // Using the method GetARandomUrl() the user can obtain a random 
   // XrdClientUrlInfo from the array.
   //
   UrlArray urlArray;
   XrdOucString listOfMachines;
   XrdOucString proto;
   XrdOucString file;
   Info(XrdClientDebug::kHIDEBUG, "XrdClientUrlSet", "parsing: "<<urls);

   // Disentangle the protocol, if any
   int p1 = 0, p2 = STR_NPOS, left = urls.length();
   if ((p2 = urls.find("://")) != STR_NPOS) {
      proto.assign(urls, p1, p2-1);
      p1 = p2 + 3;
      left = urls.length() - p1;
   }
   Info(XrdClientDebug::kHIDEBUG,"XrdClientUrlSet", "protocol: "<<proto);

   // Locate the list of machines in the input string
   if ((p2 = urls.find('/', p1)) != STR_NPOS) {
      listOfMachines.assign(urls, p1, p2-1);
      p1 = p2+1;
      left = urls.length() - p1;
   } else {
      listOfMachines.assign(urls, p1);
      left = 0;
   }

   // Nothing to do, if an empty list was found
   if (listOfMachines.length() <= 0) {
      Error("XrdClientUrlSet", "list of hosts, ports is empty" );
      fIsValid = FALSE;
      return;
   }

   // Get pathfile
   if (left > 0)
     file.assign(urls, p1);
   Info(XrdClientDebug::kHIDEBUG,"XrdClientUrlSet", "file: "<<file);

   // Init of the random number generator
   fSeed = getpid();

   //
   // We assume the protol is "root://", because this 
   // must be the protocol for
   //
   if ( (proto != "xroot") && ( proto != "root" ) ) {
      Error("XrdClientUrlSet", "This is not a root or xroot protocol: "<<proto );
      fIsValid = FALSE;
      return;
   }

   // remove trailing "," that would introduce a null host
   while ((listOfMachines.endswith(',')) || (listOfMachines.endswith(' ')))
      listOfMachines.erasefromend(1);

   // remove leading "," that would introduce a null host
   while (listOfMachines.beginswith(','))
      listOfMachines.erasefromstart(1);

   Info(XrdClientDebug::kHIDEBUG,"XrdClientUrlSet",
        "list of [host:port] : "<<listOfMachines);

   //
   // Set fPathName
   //
   fPathName = file;

   // If at this point we have a strange pathfile, then it's bad
   if ( (fPathName.length() <= 1) || (fPathName == "/") ) {
      Error("XrdClientUrlSet", "malformed pathfile " << fPathName);
      fIsValid = FALSE;
      return;
   }

   Info(XrdClientDebug::kHIDEBUG, "XrdClientUrlSet", "Remote file to open is '" <<
	fPathName << "'");
 
   if (fIsValid) {
      //
      // Parse list
      XrdOucString entity;
      int from = 0;
      while ((from = listOfMachines.tokenize(entity, from, ',')) != STR_NPOS) {
         // Convert to UrlInfo format
         Info(XrdClientDebug::kDUMPDEBUG,"XrdClientUrlSet",
                                         "parsing entity: "<<entity);
         ConvertDNSAlias(fUrlArray, proto, entity, file);
      }

      if (fUrlArray.GetSize() <= 0)
         fIsValid = FALSE;

      if (XrdClientDebug::Instance()->GetDebugLevel() >=
          XrdClientDebug::kUSERDEBUG)
         ShowUrls();
   }

}

//_____________________________________________________________________________
XrdClientUrlSet::~XrdClientUrlSet()
{
   fTmpUrlArray.Clear();

   for( int i=0; i < fUrlArray.GetSize(); i++)
      delete fUrlArray[i];

   fUrlArray.Clear();
}

//_____________________________________________________________________________
XrdClientUrlInfo *XrdClientUrlSet::GetNextUrl()
{
   // Returns the next url object pointer in the array.
   // After the last object is returned, the array is rewind-ed.
   // Now implemented as a pick from the tmpUrlArray queue

   XrdClientUrlInfo *retval;

   if ( !fTmpUrlArray.GetSize() ) Rewind();

   retval = fTmpUrlArray.Pop_back();

   return retval;
}

//_____________________________________________________________________________
void XrdClientUrlSet::Rewind()
{
   // Rebuilds tmpUrlArray, i..e the urls that have to be picked
   fTmpUrlArray.Clear();

   for(int i=0; i <= fUrlArray.GetSize()-1; i++)
      fTmpUrlArray.Push_back( fUrlArray[i] );
}

//_____________________________________________________________________________
XrdClientUrlInfo *XrdClientUrlSet::GetARandomUrl()
{
   XrdClientUrlInfo *retval;
   int rnd;

   if (!fTmpUrlArray.GetSize()) Rewind();

   // If the urlarray is still empty, just exits
   if (!fTmpUrlArray.GetSize()) return 0;
   
   for (int i=0; i < 10; i++)
      rnd = static_cast<int>(GetRandom() * fTmpUrlArray.GetSize()) % fTmpUrlArray.GetSize();

   // Returns a random url from the ones that have to be picked
   // When all the urls have been picked, we restart from the full url set

   retval = fTmpUrlArray[rnd];
   fTmpUrlArray.Erase(rnd);

   return retval;
}


//_____________________________________________________________________________
XrdClientUrlInfo *XrdClientUrlSet::GetARandomUrl(unsigned int seed)
{
   XrdClientUrlInfo *retval;

   if (!fTmpUrlArray.GetSize()) Rewind();

   // If the urlarray is still empty, just exits
   if (!fTmpUrlArray.GetSize()) return 0;
   
   // When all the urls have been picked, we restart from the full url set

   int rnd = seed % fTmpUrlArray.GetSize();
   retval = fTmpUrlArray[rnd];
   fTmpUrlArray.Erase(rnd);

   return retval;
}

//_____________________________________________________________________________
void XrdClientUrlSet::EraseUrl(XrdClientUrlInfo *url)
{
   // Eliminates url from the list

   for(int i=0; i < fUrlArray.GetSize(); i++) {
      if (url == fUrlArray[i]) {
         fUrlArray.Erase(i);
         Info(XrdClientDebug::kHIDEBUG, "EraseUrl",
                                        " url found and dropped from the list");
         return;
      }
   }
   Info(XrdClientDebug::kHIDEBUG, "EraseUrl", " url NOT found in the list");
}

//_____________________________________________________________________________
void XrdClientUrlSet::ShowUrls()
{
   // Prints the list of urls

   Info(XrdClientDebug::kUSERDEBUG, "ShowUrls",
	"The converted URLs count is " << fUrlArray.GetSize() );

   for(int i=0; i < fUrlArray.GetSize(); i++)
      Info(XrdClientDebug::kUSERDEBUG, "ShowUrls",
	   "URL n." << i+1 << ": "<< fUrlArray[i]->GetUrl() << "."); 

}

//_____________________________________________________________________________
void XrdClientUrlSet::CheckPort(int &port)
{
   // Checks the validity of port in the given host[:port]
   // Eventually completes the port if specified in the services file

   if (port <= 0) {

      // Port not specified
      Info(XrdClientDebug::kHIDEBUG, "CheckPort",
           "TCP port not specified: trying /etc/services ...");

      struct servent *svc = getservbyname("rootd", "tcp");

      if (svc <= 0) {
         Info(XrdClientDebug::kHIDEBUG, "CheckPort",
	      "service rootd not specified in /etc/services;" <<
              "using default IANA tcp port 1094");
	 port= 1094;

      } else {
	 port = ntohs(svc->s_port);
         Info(XrdClientDebug::kHIDEBUG, "CheckPort",
              "found tcp port " <<  port << "."); 
      }

   } else
      // Port is potentially valid
      Info(XrdClientDebug::kHIDEBUG, "CheckPort",
          "specified port (" <<  port << ") potentially valid."); 
}

//_____________________________________________________________________________
void XrdClientUrlSet::ConvertDNSAlias(UrlArray& urls, XrdOucString proto, 
                                      XrdOucString host, XrdOucString file)
{
   // Create an XrdClientUrlInfo from protocol 'proto', remote host 'host',
   // file 'file' and add it to the array, after having resolved the DNS
   // information.
   bool hasPort;
   XrdOucString tmpaddr;
  
   XrdClientUrlInfo *newurl = new XrdClientUrlInfo(host);
   hasPort = (newurl->Port > 0);

   if (hasPort) {
      Info(XrdClientDebug::kHIDEBUG, "ConvertDNSAlias",
           "resolving " << newurl->Host << ":" << newurl->Port);
   } else
      Info(XrdClientDebug::kHIDEBUG, "ConvertDNSAlias",
           "resolving " << newurl->Host);

   // Make sure port is a reasonable number
   CheckPort(newurl->Port);

   // Resolv the DNS information
   char *haddr[10] = {0}, *hname[10] = {0};
   int naddr = XrdNetDNS::getAddrName(newurl->Host.c_str(), 10, haddr, hname);

   // Fill the list
   int i = 0;
   for (; i < naddr; i++ ) {

      // Address
      newurl->HostAddr = (const char *) haddr[i];

      // Name
      newurl->Host = (const char *) hname[i];

      // Protocol
      newurl->Proto = proto;

      // File
      newurl->File = file;

      // Add to the list
      urls.Push_back(newurl);
      
      // Notify
      Info(XrdClientDebug::kHIDEBUG, "ConvertDNSAlias",
          "found host " << newurl->Host << " with addr " << newurl->HostAddr);

      // Get a copy, if we need to store another 
      if (i < (naddr-1))
         newurl = new XrdClientUrlInfo(*newurl);

      // Cleanup 
      if (haddr[i]) free(haddr[i]);
      if (hname[i]) free(hname[i]);
   }
}
