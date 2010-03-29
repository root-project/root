// $Id$

const char *XrdSecgsiProxyCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                   X r d S e c g s i P r o x y . c c                        */
/*                                                                            */
/* (c) 2005, G. Ganis / CERN                                                  */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Manage GSI proxies                                                         */
/*                                                                            */
/* ************************************************************************** */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>
#include <time.h>


#include <XrdOuc/XrdOucString.hh>
#include <XrdSys/XrdSysLogger.hh>
#include <XrdSys/XrdSysError.hh>

#include <XrdSut/XrdSutAux.hh>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoFactory.hh>
#include <XrdCrypto/XrdCryptoX509.hh>
#include <XrdCrypto/XrdCryptoX509Req.hh>
#include <XrdCrypto/XrdCryptoX509Chain.hh>
#include <XrdCrypto/XrdCryptoX509Crl.hh>

#include <XrdCrypto/XrdCryptosslgsiX509Chain.hh>
#include <XrdCrypto/XrdCryptosslgsiAux.hh>

#include <XrdSecgsi/XrdSecgsiTrace.hh>

#define PRT(x) {cerr <<x <<endl;}

//
// enum
enum kModes {
   kM_undef = 0,
   kM_init = 1,
   kM_info,
   kM_destroy,
   kM_help
};
const char *gModesStr[] = {
   "kM_undef",
   "kM_init",
   "kM_info",
   "kM_destroy",
   "kM_help"
};

//
// Prototypes
//
void Menu();
int  ParseArguments(int argc, char **argv);
bool CheckOption(XrdOucString opt, const char *ref, int &ival);
void Display(XrdCryptoX509 *xp);

//
// Globals 
//
int          Mode     = kM_undef;
bool         Debug = 0;
bool         Exists = 0;
XrdCryptoFactory *gCryptoFactory = 0;
XrdOucString CryptoMod = "ssl";
XrdOucString CAdir  = "/etc/grid-security/certificates/";
XrdOucString CRLdir = "/etc/grid-security/certificates/";
XrdOucString DefEEcert = "/.globus/usercert.pem";
XrdOucString DefEEkey  = "/.globus/userkey.pem";
XrdOucString DefPXcert = "/tmp/x509up_u";
XrdOucString EEcert = "";
XrdOucString EEkey  = "";
XrdOucString PXcert = "";
XrdOucString Valid  = "12:00";
int          Bits   = 512;
int          PathLength = 0;
int          ClockSkew = 30;
// For error logging and tracing
static XrdSysLogger Logger;
static XrdSysError eDest(0,"proxy_");
XrdOucTrace *gsiTrace = 0;

int main( int argc, char **argv )
{
   // Test implemented functionality
   int secValid = 0;
   XrdProxyOpt_t pxopt;
   XrdCryptosslgsiX509Chain *cPXp = 0;
   XrdCryptoX509 *xPXp = 0;
   XrdCryptoRSA *kPXp = 0;
   XrdCryptoX509ParseFile_t ParseFile = 0;
   int prc = 0;
   int nci = 0;
   int exitrc = 0;

   // Parse arguments
   if (ParseArguments(argc,argv)) {
      exit(1);
   }

   //
   // Initiate error logging and tracing
   eDest.logger(&Logger);
   if (!gsiTrace)
      gsiTrace = new XrdOucTrace(&eDest);
   if (gsiTrace) {
      if (Debug)
        // Medium level
        gsiTrace->What |= (TRACE_Authen | TRACE_Debug);
   }
   //
   // Set debug flags in other modules
   if (Debug) {
      XrdSutSetTrace(sutTRACE_Debug);
      XrdCryptoSetTrace(cryptoTRACE_Debug);
   }

   //
   // Load the crypto factory
   if (!(gCryptoFactory = XrdCryptoFactory::GetCryptoFactory(CryptoMod.c_str()))) {
      PRT(": cannot instantiate factory "<<CryptoMod);
      exit(1);
   }
   if (Debug)
      gCryptoFactory->SetTrace(cryptoTRACE_Debug);

   //
   // Depending on the mode
   switch (Mode) {
   case kM_help:
      //
      // We should not get here ... print the menu and go
      Menu();
      break;
   case kM_init:
      //
      // Init proxies
      secValid = XrdSutParseTime(Valid.c_str(), 1);
      pxopt.bits = Bits;
      pxopt.valid = secValid;
      pxopt.depthlen = PathLength;
      cPXp = new XrdCryptosslgsiX509Chain();
      prc = XrdSslgsiX509CreateProxy(EEcert.c_str(), EEkey.c_str(), &pxopt,
                                     cPXp, &kPXp, PXcert.c_str());
      if (prc == 0) {
         // The proxy is the first certificate
         xPXp = cPXp->Begin();
         if (xPXp) {
            Display(xPXp);
         } else {
            PRT( ": proxy certificate not found");
         }
      } else {
         PRT( ": problems creating proxy");
      }
      break;
   case kM_destroy:
      //
      // Destroy existing proxies
      if (unlink(PXcert.c_str()) == -1) {
         perror("xrdgsiproxy");
      }

      break;
   case kM_info:
      //
      // Display info about existing proxies
      if (!(ParseFile = gCryptoFactory->X509ParseFile())) {
         PRT("cannot attach to ParseFile function!");
         break;
      }
      // Parse the proxy file
      cPXp = new XrdCryptosslgsiX509Chain();
      nci = (*ParseFile)(PXcert.c_str(), cPXp);
      if (nci < 2) {
         if (Exists) {
            exitrc = 1;
         } else {
            PRT("proxy files must have at least two certificates"
                " (found only: "<<nci<<")");
         }
         break;
      }
      // The proxy is the first certificate
      xPXp = cPXp->Begin();
      if (xPXp) {
         if (!Exists) {
            Display(xPXp);
         } else {
            // Check time validity
            secValid = XrdSutParseTime(Valid.c_str(), 1);
            int tl = xPXp->NotAfter() -(int)time(0);
            if (Debug)
               PRT("secValid: " << secValid<< ", tl: "<<tl<<", ClockSkew:"<<ClockSkew);
            if (secValid > tl + ClockSkew) {
               exitrc = 1;
               break;
            }
            // Check bit strenght
            if (Debug)
               PRT("BitStrength: " << xPXp->BitStrength()<< ", Bits: "<<Bits);
            if (xPXp->BitStrength() < Bits) {
               exitrc = 1;
               break;
            }
         }
      } else {
         if (Exists) {
            exitrc = 1;
         } else {
            PRT( ": proxy certificate not found");
         }
      }
      break;
   default:
      //
      // Print menu
      Menu();
   }

   exit(exitrc);
}

int ParseArguments(int argc, char **argv)
{
   // Parse application arguments filling relevant global variables

   // Number of arguments
   if (argc < 0 || !argv[0]) {
      PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
      PRT("+ Insufficient number or arguments!                        +");
      PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
      // Print main menu
      Menu();
      return 1;
   }
   --argc;
   ++argv;

   //
   // Loop over arguments
   while ((argc >= 0) && (*argv)) {

      XrdOucString opt = "";
      int ival = -1;
      if(*(argv)[0] == '-') {

         opt = *argv;
         opt.erase(0,1); 
         if (CheckOption(opt,"h",ival) || CheckOption(opt,"help",ival) ||
             CheckOption(opt,"menu",ival)) {
            Mode = kM_help;
         } else if (CheckOption(opt,"debug",ival)) {
            Debug = ival;
         } else if (CheckOption(opt,"e",ival)) {
            Exists = 1;
         } else if (CheckOption(opt,"exists",ival)) {
            Exists = 1;
         } else if (CheckOption(opt,"f",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               PXcert = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-f' requires a proxy file name: ignoring         +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"file",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               PXcert = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-file' requires a proxy file name: ignoring      +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"out",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               PXcert = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-out' requires a proxy file name: ignoring        +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"cert",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               EEcert = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-cert' requires a cert file name: ignoring       +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"key",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               EEkey = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-key' requires a key file name: ignoring         +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"certdir",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               CAdir = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-certdir' requires a dir path: ignoring          +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"valid",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               Valid = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-valid' requires a time string: ignoring         +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"path-length",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               PathLength = strtol(*argv,0,10);
               if (PathLength < -1 || errno == ERANGE) {
                  PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                  PRT("+ Option '-path-length' requires a number >= -1: ignoring  +");
                  PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                  argc++;
                  argv--;
               }
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-path-length' requires a number >= -1: ignoring  +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"bits",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               Bits = strtol(*argv, 0, 10);
               Bits = (Bits > 512) ? Bits : 512;
               if (errno == ERANGE) {
                  PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                  PRT("+ Option '-bits' requires a number: ignoring               +");
                  PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                  argc++;
                  argv--;
               }
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-bits' requires a number: ignoring               +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"clockskew",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               ClockSkew = strtol(*argv, 0, 10);
               if (ClockSkew < -1 || errno == ERANGE) {
                  PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                  PRT("+ Option '-clockskew' requires a number >= -1: ignoring    +");
                  PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                  argc++;
                  argv--;
               }
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-clockskew' requires a number >= -1: ignoring    +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Ignoring unrecognized option: "<<*argv);
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         }

      } else {
         //
         // Mode keyword
         opt = *argv;
         if (CheckOption(opt,"init",ival)) {
            Mode = kM_init;
         } else if (CheckOption(opt,"info",ival)) {
            Mode = kM_info;
         } else if (CheckOption(opt,"destroy",ival)) {
            Mode = kM_destroy;
         } else if (CheckOption(opt,"help",ival)) {
            Mode = kM_help;
         } else {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Ignoring unrecognized keyword mode: "<<opt.c_str());
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         }
      }
      --argc;
      ++argv;
   }

   //
   // Default mode 'info'
   Mode = (Mode == 0) ? kM_info : Mode;

   //
   // If help mode, print menu and exit
   if (Mode == kM_help) {
      // Print main menu
      Menu();
      return 1;
   }

   //
   // we may need later
   struct passwd *pw = 0;

   //
   // Check proxy file
   if (PXcert.length() <= 0) {
      // Use defaults
      if (!pw && !(pw = getpwuid(getuid()))) {
         // Cannot get info about current user
         PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         PRT("+ Cannot get info about current user - exit ");
         PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         return 1;
      }
      // Build proxy file name
      PXcert = DefPXcert + (int)(pw->pw_uid);
   }
   //
   // Expand Path
   XrdSutExpand(PXcert);
   // Get info
   struct stat st;
   if (stat(PXcert.c_str(),&st) != 0) {
      if (errno != ENOENT) {
         // Path exists but we cannot access it - exit
         PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         PRT("+ Cannot access requested proxy file: "<<PXcert.c_str());
         PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         return 1;
      } else {
         if (Mode != kM_init) {
            // Path exists but we cannot access it - exit
            if (!Exists) {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ proxy file: "<<PXcert.c_str()<<" not found");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            }
            return 1;
         }
      }
   }

   //
   // The following applies in 'init' mode only
   if (Mode == kM_init) {
      //
      // Check certificate file
      if (EEcert.length()) {
         //
         // Expand Path
         XrdSutExpand(EEcert);
         // Get info
         if (stat(EEcert.c_str(),&st) != 0) {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Cannot access certificate file: "<<EEcert.c_str());
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            return 1;
         }
      } else {
         // Use defaults
         if (!pw && !(pw = getpwuid(getuid()))) {
            // Cannot get info about current user
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Cannot get info about current user - exit ");
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            return 1;
         }
         EEcert = DefEEcert;
         EEcert.insert(XrdSutHome(), 0);
         if (stat(EEcert.c_str(),&st) != 0) {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Cannot access certificate file: "<<EEcert.c_str());
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            return 1;
         }
      }
      //
      // Check key file
      if (EEkey.length()) {
         //
         // Expand Path
         XrdSutExpand(EEkey);
         // Get info
         if (stat(EEkey.c_str(),&st) != 0) {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Cannot access private key file: "<<EEkey.c_str());
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            return 1;
         }
      } else {
         // Use defaults
         if (!pw && !(pw = getpwuid(getuid()))) {
            // Cannot get info about current user
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Cannot get info about current user - exit ");
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            return 1;
         }
         EEkey = DefEEkey;
         EEkey.insert(XrdSutHome(), 0);
         if (stat(EEkey.c_str(),&st) != 0) {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Cannot access certificate file: "<<EEkey.c_str());
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            return 1;
         }
      }
      // Check permissions
      if (!S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) ||
          (st.st_mode & (S_IWGRP | S_IWOTH)) != 0 ||
          (st.st_mode & (S_IRGRP | S_IROTH)) != 0 ||
          (st.st_mode & (S_IWUSR)) != 0) {
          PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
          PRT("+ Wrong permissions for file: "<<EEkey.c_str());
          PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
          return 1;
      }
   }

   return 0;
}



void Menu()
{
   // Print the menu

   PRT(" ");
   PRT(" xrdgsiproxy: application to manage GSI proxies ");
   PRT(" ");
   PRT(" Syntax:");
   PRT(" ");
   PRT(" xrdgsiproxy [-h] [<mode>] [options] ");
   PRT(" ");
   PRT(" ");
   PRT("  -h   display this menu");
   PRT(" ");
   PRT(" mode (info, init, destroy) [info]");
   PRT(" ");
   PRT("       info: display content of existing proxy file");
   PRT(" ");
   PRT("       init: create proxy certificate and related proxy file");
   PRT(" ");
   PRT("       destroy: delete existing proxy file");
   PRT(" ");
   PRT(" options:");
   PRT(" ");
   PRT("    -debug                 Print more information while running this"
                                   " query (use if something goes wrong) ");
   PRT(" ");
   PRT("    -f,-file,-out <file>   Non-standard location of proxy file");
   PRT(" ");
   PRT("    init mode only:");
   PRT(" ");
   PRT("    -certdir  <dir>        Non-standard location of directory"
                                   " with information about known CAs");
   PRT("    -cert     <file>       Non-standard location of certificate"
                                   " for which proxies are wanted");
   PRT("    -key      <file>       Non-standard location of the private"
                                   " key to be used to sign the proxy");
   PRT("    -bits     <bits>       strength in bits of the key [512]");
   PRT("    -valid    <hh:mm>      Time validity of the proxy certificate [12:00]");
   PRT("    -path-length <len>     max number of descendent levels below"
                                   " this proxy [0] ");
   PRT("    -e,-exists [options]   returns 0 if valid proxy exists, 1 otherwise;");
   PRT("                           valid options: '-valid <hh:mm>', -bits <bits>");
   PRT("    -clockskew <secs>      max clock-skewness allowed when checking time validity [30 secs]");
   PRT(" ");
}

bool CheckOption(XrdOucString opt, const char *ref, int &ival)
{
   // Check opt against ref
   // Return 1 if ok, 0 if not
   // Fills ival = 1 if match is exact
   //       ival = 0 if match is exact with no<ref> 
   //       ival = -1 in the other cases
   bool rc = 0;

   int lref = (ref) ? strlen(ref) : 0;
   if (!lref) 
      return rc;
   XrdOucString noref = ref;
   noref.insert("no",0);

   ival = -1;
   if (opt == ref) {
      ival = 1;
      rc = 1;
   } else if (opt == noref) {
      ival = 0;
      rc = 1;
   }

   return rc;
}

void Display(XrdCryptoX509 *xp)
{
   // display content of proxy certificate

   PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
   if (!xp) {
      PRT(" Empty certificate! ");
      return;
   }

   // File
   PRT("file        : "<<PXcert);
   // Issuer
   PRT("issuer      : "<<xp->Issuer());
   // Subject
   PRT("subject     : "<<xp->Subject());
   // Path length field
   int pathlen = 0;
   XrdSslgsiProxyCertInfo(xp->GetExtension(gsiProxyCertInfo_OID), pathlen);
   PRT("path length : "<<pathlen);
   // Key strength
   PRT("bits        : "<<xp->BitStrength());
   // Time left
   int tl = xp->NotAfter() -(int)time(0);
   int hh = (tl >= 3600) ? (tl/3600) : 0; tl -= (hh*3600); 
   int mm = (tl >= 60)   ? (tl/60)   : 0; tl -= (mm*60); 
   int ss = (tl >= 0)    ?  tl       : 0; 
   PRT("time left   : "<<hh<<"h:"<<mm<<"m:"<<ss<<"s");
   PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
}
