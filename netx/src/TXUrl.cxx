/**********************************************************************/
/*                          T X U r l . c c                           */
/*                                  2003                              */
/*   Produced by Alvise Dorigo & Fabrizio Furano for INFN padova      */
/**********************************************************************/
//
//   $Id: TXUrl.cc,v 1.7 2004/06/30 15:48:45 furano Exp $
//
// Author: Alvise Dorigo, Fabrizio Furano

#include "TXUrl.h"
#include "TXDebug.h"
#include "TString.h"
#include "TError.h"
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <ctype.h>               // needed by isdigit()
#include <netdb.h>               // needed by getservbyname()
#include <netinet/in.h>          // needed by ntohs()
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <resolv.h>
#include <arpa/nameser.h>
#include <sys/time.h>
#include <unistd.h>
#include "TSystem.h"
#include "TInetAddress.h"

extern int h_errno;

using namespace std;

ClassImp(TXUrl);



//_____________________________________________________________________________
TXUrl::TXUrl(TString Urls) : fIsValid(kTRUE)
{
   // A container for multiple urls.
   // It creates an array of multiple urls parsing the argument Urls and
   //  resolving the DNS aliases
   //
   // Urls MUST be in the form:
   //
   //    root://[username1@]server1:port1[,[username2@]server2:port2, ... ,
   //           [usernameN@]serverN:portN]/pathfile
   //
   // Using the method GetNextUrl() the user can obtain the next TUrl object pointer in the array
   // (the array is cyclic).
   // Using the method GetARandomUrl() the user can obtain a random TUrl from the array

   UrlArray urlArray;
   TString listOfMachines;
   TString protoToMatch(PROTO);
  
   protoToMatch += "://";
  
   // Init of the random number generator with the internal clock as a seed
   fRndgen.SetSeed(gSystem->GetPid());

   //
   // We assume the protol is "root://", because this 
   // must be the protocol for TXNetFile
   //


   if ( !Urls.BeginsWith(protoToMatch) ) {
      Error("TXUrl", "This is not a %s protocol.", protoToMatch.Data() );
      fIsValid = kFALSE;
   } else
      // remove leading 'root://' from the string
      // Now sUrls will contain [user1@]machine1[:port1],...,
      //                        [userN]machineN[:portN]/pathfile
      Urls.Remove( 0, protoToMatch.Length() ); 

   //
   // Save the list of comma separated servers:ports, assuming they are
   // separated by a '/' char from the pathfile
   //
   listOfMachines = Urls;

   Short_t slashPos = (Short_t)Urls.First('/');
   if( slashPos != kNPOS) {
      listOfMachines.Remove(slashPos);
   }

   // remove trailing "," that would introduce a null host
   while (listOfMachines.EndsWith(","))
      listOfMachines.Remove(listOfMachines.Length()-1);

   // remove leading "," that would introduce a null host
   while (listOfMachines.BeginsWith(","))
      listOfMachines.Remove(0,1);

   if(DebugLevel() >= TXDebug::kUSERDEBUG)
      Info("TXUrl", "List of servers to connect to is [%s]",
                    listOfMachines.Data());

   //
   // Set fPathName
   //
   fPathName = "";

   if( slashPos != kNPOS) {
      fPathName = Urls;
      fPathName.Remove(0, slashPos);
   }

   // If at this point we have a strange pathfile, then it's bad
   if ( fPathName.CompareTo("/") == 0 ) {
      Error("TXUrl", "Malformed pathfile %s", fPathName.Data());
      fIsValid = kFALSE;
   }

   if(DebugLevel() >= TXDebug::kHIDEBUG)
      Info("TXUrl", "Remote file to open is [%s]", fPathName.Data());
 
   if (fIsValid) {
      ConvertDNSAliases(fUrlArray, listOfMachines, fPathName);

      if (fUrlArray.size() <= 0)
	 fIsValid = kFALSE;

      if(DebugLevel() >= TXDebug::kUSERDEBUG)
	 ShowUrls();
   }

}

//_____________________________________________________________________________
TXUrl::~TXUrl()
{
   fTmpUrlArray.clear();

   for( UShort_t i=0; i < fUrlArray.size(); i++)
      SafeDelete( fUrlArray[i] );

   fUrlArray.clear();
}

//_____________________________________________________________________________
TUrl *TXUrl::GetNextUrl()
{
   // Returns the next TUrl object pointer in the array.
   // After the last object is returned, the array is rewind-ed.
   // Now implemented as a pick from the tmpUrlArray queue

   TUrl *retval;

   if ( !fTmpUrlArray.size() ) Rewind();

   retval = fTmpUrlArray.back();

   fTmpUrlArray.pop_back();

   return retval;
}

//_____________________________________________________________________________
void TXUrl::Rewind()
{
   // Rebuilds tmpUrlArray, i..e the urls that have to be picked
   fTmpUrlArray.clear();

   for( UShort_t i=0; i <= fUrlArray.size()-1; i++)
      fTmpUrlArray.push_back( fUrlArray[i] );
}

//_____________________________________________________________________________
TUrl *TXUrl::GetARandomUrl()
{
   TUrl *retval;
   UInt_t rnd;

   for (int i=0; i < 10; i++)
      rnd = fRndgen.Integer(fTmpUrlArray.size());

   // Returns a random url from the ones that have to be picked
   // When all the urls have been picked, we restart from the full url set

   if ( !fTmpUrlArray.size() ) Rewind();

   UrlArray::iterator it = fTmpUrlArray.begin() + rnd;

   retval = *it;
   fTmpUrlArray.erase(it);

   return retval;
}

//_____________________________________________________________________________
void TXUrl::ShowUrls()
{
   // Prints the list of urls

   Info("ShowUrls", "The converted URLs count is %d.", fUrlArray.size());

   for(UInt_t i=0; i < fUrlArray.size(); i++)
      Info("ShowUrls", "URL n.%d: %s .", i+1, fUrlArray[i]->GetUrl()); 

}

//_____________________________________________________________________________
void TXUrl::CheckPort(TString &machine)
{
   // Checks the validity of port in the given host[:port]
   // Eventually completes the port if specified in the services file
   Short_t commaPos = machine.First(':');

   if(commaPos == kNPOS) {
      // Port not specified

      if(DebugLevel() >= TXDebug::kHIDEBUG)
	 Warning("checkPort", 
		 "TCP port not specified for host %s. Trying to get it from /etc/services...", machine.Data());

      Int_t prt = gSystem->GetServiceByName("rootd");

      if(prt <= 0) {
	 if(DebugLevel() >= TXDebug::kHIDEBUG)
	    Warning("checkPort", "Service %s not specified in /etc/services; using default IANA tcp port 1094", PROTO);
	 machine += ":1094";
      } else {
         if (DebugLevel() >= TXDebug::kHIDEBUG)
     	    Info("checkPort", "Found tcp port %d in /etc/service", prt);
	 
	 machine += ":";
	 machine += prt;
      }

   } else {
      // The port seems there

      TString tmp(machine);
      tmp.Remove(0, commaPos+1);

      TString sPort = tmp;

      if(sPort.CompareTo("") == 0)
	 Error("checkPort","The specified tcp port is 0 for %s", machine.Data());
      else {
	 for(Short_t i=0; i<=sPort.Length()-1; i++)
	    if(isdigit(sPort[i]) == 0) {
	       Error("checkPort","The specified tcp port is not numeric for %s", machine.Data());
	    }
      }
   }
}

//_____________________________________________________________________________
Bool_t TXUrl::ConvertSingleDNSAlias(UrlArray& urls, TString hostname, 
                                                    TString fname)
{
   // Converts a single host[:port] into an array of TUrl
   // The new Turls are appended to the given UrlArray


   Bool_t specifiedPort;
   Int_t port = 0;


   TString tmpaddr;
  
   specifiedPort = ( hostname.First(':') != kNPOS );
  
   hostname.Append("/fakefile");
   TUrl tmp(hostname.Data());

   if(specifiedPort) {
      port = tmp.GetPort();
      if(DebugLevel() >= TXDebug::kHIDEBUG)
	 Info("ConvertSingleDNSAlias","Resolving %s:%d.", tmp.GetHost(), port);
   } else
      if(DebugLevel() >= TXDebug::kHIDEBUG)
	 Info("ConvertSingleDNSAlias","Resolving %s.", tmp.GetHost());

   TInetAddress iaddr = gSystem->GetHostByName(tmp.GetHost());

   if(!iaddr.IsValid()) {
      Error("ConvertSingleDNSAlias","GetHostByName error for host %s. %s",
	    tmp.GetHost(), strerror(h_errno));
      return kFALSE;
   }

   // Let's get the list of the addresses for the host
   TInetAddress::AddressList_t::const_iterator j = 
                                    iaddr.GetAddresses().begin();
   for ( ; j != iaddr.GetAddresses().end(); j++) {

      const char *addr = TInetAddress::GetHostAddress(*j);
      TInetAddress c = gSystem->GetHostByName(addr);

      // Set the user for the child urls
      tmpaddr = "";
      if ( strlen(tmp.GetUser()) ) {
	 tmpaddr = tmp.GetUser();
	 tmpaddr += "@";
      }

      if (!strcmp(c.GetHostName(), "UnNamedHost"))
	 tmpaddr += addr;
      else
	 tmpaddr += c.GetHostName();
      
      if(DebugLevel() >= TXDebug::kHIDEBUG)
	 Info("ConvertSingleDNSAlias","Found host %s", tmpaddr.Data() );

      if(specifiedPort) {
	 char Port[7];
	 memset((void *)Port, 0, 7);
	 sprintf((char *)Port, "%d", port);
	 tmpaddr.Append(':');
	 tmpaddr.Append(Port);
      }

      if (fname.Length())
	 tmpaddr.Append(fname);

      urls.push_back( new TUrl(tmpaddr.Data()) );
   }

   return (urls.size() > 0);
}

//_____________________________________________________________________________
void TXUrl::ConvertDNSAliases(UrlArray& urls, TString list, TString fname)
{
   // Given a list of comma-separated host[:port]
   // every entry is resolved via DNS into its aliases
   // The parameter is overwritten with the processed data
   Short_t colonPos;

   list += ",";

   while(list.Length() > 0) {
      colonPos = list.First(',');
      if (colonPos != kNPOS) {
	 TString tmp(list);

	 tmp.Remove(colonPos);
	 list.Remove(0,colonPos+1);

	 CheckPort(tmp);
	 ConvertSingleDNSAlias(urls, tmp, fname);
      }

   }


      
}

//_____________________________________________________________________________
const char *TXUrl::ConvertIP_to_Name(const char* IP) 
{
   struct in_addr A;
   struct hostent *H;
   //  char str[INET_ADDRSTRLEN];

   memset((void *)&A, 0, sizeof(A));
   inet_pton(AF_INET, IP, (void *)&A);
   //  H = gethostbyaddr(&A, sizeof(A), AF_INET);
   H = gethostbyaddr((const char *)&A, sizeof(A), AF_INET);
   //  cout << H->h_name << endl;

   if (H)
      return (const char* )H->h_name;
   else
      return IP;
}


