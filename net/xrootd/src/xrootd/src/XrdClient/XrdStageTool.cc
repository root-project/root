//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdStagetool                                                         //
//                                                                      //
// Author: Fabrizio Furano (CERN, 2007)                                 //
//                                                                      //
// A command line tool for xrootd environments, to trigger sync or async//
// staging of files                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

const char *XrdStageToolCVSID = "$Id$";

#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>
#include <sstream>

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#include <curses.h>
#include <term.h>
#endif


/////////////////////////////////////////////////////////////////////
// function + macro to allow formatted print via cout,cerr
/////////////////////////////////////////////////////////////////////
extern "C" {

   void cout_print(const char *format, ...) {
      char cout_buff[4096];
      va_list args;
      va_start(args, format);
      vsprintf(cout_buff, format,  args);
      va_end(args);
      cout << cout_buff;
   }

   void cerr_print(const char *format, ...) {
      char cerr_buff[4096];
      va_list args;
      va_start(args, format);
      vsprintf(cerr_buff, format,  args);
      va_end(args);
      cerr << cerr_buff;
   }

#define COUT(s) do {				\
      cout_print s;				\
   } while (0)

#define CERR(s) do {				\
      cerr_print s;				\
   } while (0)

}

//////////////////////////////////////////////////////////////////////


#define XRDSTAGETOOL_VERSION            "(C) 2004-2010 by the Xrootd group. $Revision: 1.9 $ - Xrootd version: "XrdVSTRING


///////////////////////////////////////////////////////////////////////
// Coming from parameters on the cmd line

XrdOucString opaqueinfo;

// Default open flags for opening a file (xrd)
kXR_unt16 xrd_open_flags = kXR_retstat;

XrdOucString srcurl;
bool useprepare = false;
int maxopens = 2;
int dbglvl = 0;
int verboselvl = 0;

///////////////////////




///////////////////////////////////////////////////////////////////////
// Generic instances used throughout the code

XrdClient *genclient;
XrdClientAdmin *genadmin = 0;
XrdClientVector<XrdClientUrlInfo> urls;

struct OpenInfo {
  XrdClient *cli;
  XrdClientUrlInfo *origurl;
};

XrdClientVector<struct OpenInfo> opening;

///////////////////////

void PrintUsage() {
   cerr << 
     "usage1: xrdstagetool [-d dbglevel] [-p] [-s] [-O info] xrootd_url1 ... xrootd_urlN " << endl <<
     " Requests xrootd MSS staging for the listed complete xrootd URLs" << endl <<
     "  in the form root://host//absolute_path_of_a_file" << endl <<
     "  Please note that in the xrootd world a MSS is not necessarily a tape system. " << endl <<
     "  Some form of staging system must be set up in each contacted host." << endl << endl <<
     "usage2: xrdstagetool [-d dbglevel] [-p] xrootd_url_dest -S xrootd_url_src" << endl <<
     " Contacts the dest host and requests to stage the file xrootd_url_dest" << endl <<
     "  by fetching it from xrootd_url_src." <<
     " This feature must be set up in the dest host, and the src host must be reachable by dest host."<< endl <<
     "  and by all its data servers." << endl << endl <<
     " Parameters:" << endl <<
     "  -d dbglevel      : set the XrdClient debug level (0..4)" << endl <<
     "  -p               : asynchronous staging through Prepare request" << endl <<
     "                     (must be set up at the involved server side)" << endl <<
     "  -O info          : add some opaque info to the issued requests" << endl;
}


bool CheckAnswer(XrdClientAbs *gencli) {
   if (!gencli->LastServerResp()) return false;

   switch (gencli->LastServerResp()->status) {
   case kXR_ok:
      return true;

   case kXR_error:

      cout << "Error " << gencli->LastServerError()->errnum <<
	 ": " << gencli->LastServerError()->errmsg << endl << endl;
      return false;

   default:
      cout << "Server response: " << gencli->LastServerResp()->status << endl;
      return true;

   }
}


// Main program
int main(int argc, char**argv) {

  dbglvl = -1;

   // We want this tool to be able to connect everywhere
   // Note that the side effect of these calls here is to initialize the
   // XrdClient environment.
   // This is crucial if we want to later override its default values
   EnvPutString( NAME_REDIRDOMAINALLOW_RE, "*" );
   EnvPutString( NAME_CONNECTDOMAINALLOW_RE, "*" );
   EnvPutString( NAME_REDIRDOMAINDENY_RE, "" );
   EnvPutString( NAME_CONNECTDOMAINDENY_RE, "" );

   if (argc <= 1) {
     PrintUsage();
     exit(0);
   }

   for (int i=1; i < argc; i++) {

     
     if ( (strstr(argv[i], "-O") == argv[i]) && (argc >= i+2)) {
	 opaqueinfo=argv[i+1];
	 ++i;
	 continue;
      }

      if ( (strstr(argv[i], "-h") == argv[i])) {
	 PrintUsage();
	 exit(0);
      }

      if ( (strstr(argv[i], "-p") == argv[i])) {
	 // Use prepare instead of Open
 	 useprepare = true;
	 continue;
      }

      if ( (strstr(argv[i], "-v") == argv[i])) {
	 // Increase verbosity level
 	 verboselvl++;
	 cout << "Verbosity level is now " << verboselvl << endl;
	 continue;
      }

      if ( (strstr(argv[i], "-d") == argv[i])) {
	 // Debug level
	 dbglvl = atoi(argv[i+1]);
	 i++;
	 continue;
      }

      if ( (strstr(argv[i], "-S") == argv[i])) {
	 // The url to fetch the file from
	 srcurl = argv[i+1];
	 i++;
	 continue;
      }

      if ( (strstr(argv[i], "-m") == argv[i])) {
	 // Max number of concurrent open reqs
	 maxopens = atoi(argv[i+1]);
	 i++;
	 continue;
      }

      // Any other par is considered as an url and queued
      if ( (strstr(argv[i], "-") != argv[i]) && (strlen(argv[i]) > 1) ) {
	// Enqueue
	if (verboselvl > 0)
	  cout << "Enqueueing file " << argv[i] << endl;
	XrdClientUrlInfo u(argv[i]);
	urls.Push_back(u);

	if (verboselvl > 1)
	  cout << "Enqueued URL " << u.GetUrl() << endl;

	continue;
      }


   }

   EnvPutInt(NAME_DEBUG, dbglvl);
   EnvPutInt(NAME_TRANSACTIONTIMEOUT, 3600);

   if (useprepare) {
     // Fire all the prepare requests at max speed

     for (int i = 0; i < urls.GetSize(); i++) {
       XrdClientUrlInfo u;

       if (srcurl.length() > 5) {
	 // If -S is specified and has a non trivial content,
	 // we must connect to the dest host anyway
	 // but add the source url as opaque info to the filename
	 u.TakeUrl(urls[i].GetUrl().c_str());
	 u.File += "?fetchfrom=";
	 u.File += srcurl;
       }
       else u.TakeUrl(urls[i].GetUrl().c_str());

       if (opaqueinfo.length() > 0) {
	 // Take care of the heading "?"
	 if (opaqueinfo[0] != '?') {
	   u.File += "?";
	 }

	 u.File += opaqueinfo;
       }

       XrdClientAdmin adm(u.GetUrl().c_str());

       if (verboselvl > 1)
	 cout << "Connecting to: " << u.GetUrl() << endl;

       if (!adm.Connect()) {
	 cout << "Unable to connect to " << u.GetUrl() << endl;
	 continue;
       }

       if (verboselvl > 0)
	 cout << "Requesting prepare for: " << u.GetUrl() << endl;

       if (!adm.Prepare(u.File.c_str(), (kXR_char)kXR_stage, 0)) {
	 cout << "Unable to send Prepare request for " << u.GetUrl() << endl;
	 continue;
       }

     }
   }
   else
     while((urls.GetSize() > 0) || (opening.GetSize() > 0)) {
       // Open all the files in sequence, asynchronously
       // Keep a max of maxopens as outstanding

       // See if there are open instances to clean up
       for (int i = opening.GetSize()-1; (i >= 0) && (opening.GetSize() > 0); i--) {
	 struct OpenInfo oi = opening[i];

	 if ( !oi.cli->IsOpen_inprogress() ) {
	     struct XrdClientStatInfo sti;

	     if (oi.cli->IsOpen_wait() && oi.cli->Stat(&sti)) {
	       cout << "OK " << oi.origurl->GetUrl() << 
		 " Size: " << sti.size << endl;
	     }
	     else {
	       cout << "FAIL " << oi.origurl->GetUrl() << endl;
	     }

	     // In any case this element has to be removed.
	     delete oi.cli;
	     delete oi.origurl;
	     opening.Erase(i);
	 }
       }

       // See how many attempts to start now
       int todonow = maxopens - opening.GetSize();
       todonow = xrdmin(todonow, urls.GetSize());

       if (verboselvl > 1)
	 cout << "Sync staging attempts to start: " << todonow << endl;

       for (int i = 0; i < todonow; i++) {
	 XrdClient *cli = new XrdClient(urls[0].GetUrl().c_str());
	 XrdClientUrlInfo u(urls[0]);
	 urls.Erase(0);

	 if (!cli || !cli->Open(0, xrd_open_flags))
	   cerr << "Error opening '" << endl << u.GetUrl() << endl;
	 else {
	   struct OpenInfo oi;
	   oi.cli = cli;
	   oi.origurl = new XrdClientUrlInfo(u);
	   opening.Push_back(oi);
	 }
       }



       sleep(1);

     } // while
       




   cout << endl;
   return 0;

}
