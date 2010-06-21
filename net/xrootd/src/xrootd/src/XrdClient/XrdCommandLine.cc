//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdCommandLine                                                       //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2005)                          //
//                                                                      //
// A command line tool for xrootd environments. The executable normally //
// is named xrd.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

const char *XrdCommandLineCVSID = "$Id$";

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
#include <signal.h>

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



#define XRDCLI_VERSION            "(C) 2004-2010 by the Xrootd group. $Revision: 1.34 $ - Xrootd version: "XrdVSTRING


///////////////////////////////////////////////////////////////////////
// Coming from parameters on the cmd line

char *opaqueinfo = 0;   // opaque info to be added to urls

// Default open flags for opening a file (xrd)
kXR_unt16 xrd_wr_flags=kXR_async | kXR_mkpath | kXR_open_updt | kXR_new;

char *initialhost;

///////////////////////

///////////////////////////////////////////////////////////////////////
// Generic instances used throughout the code

XrdClient *genclient = 0;
XrdClientAdmin *genadmin = 0;
XrdOucString currentpath = "/";

XrdOucString  cmdline_cmd;

///////////////////////


void
CtrlCHandler(int sig) {
   cerr << endl << "Please use 'exit' to terminate this program." << endl;
   return;
}



void PrintUsage() {
   cerr << "usage: xrd [host]"

     "[-DSparmname stringvalue] ... [-DIparmname intvalue]  [-O<opaque info>] [command]" << endl;
   cerr << " -DSparmname stringvalue     :         override the default value of an internal"
      " XrdClient setting (of string type)" << endl;
   cerr << " -DIparmname intvalue        :         override the default value of an internal"
      " XrdClient setting (of int type)" << endl;
   cerr << " -O     :         adds some opaque information to any used xrootd url" << endl;
   cerr << " -h     :         this help screen" << endl;

   cerr << endl << " where:" << endl;
   cerr << "   parmname     is the name of an internal parameter" << endl;
   cerr << "   stringvalue  is a string to be assigned to an internal parameter" << endl;
   cerr << "   intvalue     is an int to be assigned to an internal parameter" << endl;
   cerr << "   command      is a command line to be executed. in this case the host is mandatory." << endl;

   cerr << endl;
}

void PrintPrompt(stringstream &s) {
  s.clear();
  if (genadmin)
    s << "root://" << genadmin->GetCurrentUrl().Host << 
      ":" << genadmin->GetCurrentUrl().Port <<
      "/" << currentpath;

  s << ">";

}

#ifndef HAVE_READLINE
// replacement function for GNU readline
char *readline(const char *prompt) {
  char *linebuf = new char[4096];

  cout << prompt;
  if (!fgets(linebuf, 4096, stdin) || !strlen(linebuf))
    return NULL;
  return linebuf;
}
#endif

void PrintHelp() {

   cout << endl <<
      XRDCLI_VERSION << endl << endl <<
      "Usage: xrd [-O<opaque_info>] [-DS<var_name> stringvalue] [-DI<var_name> integervalue] [host[:port]] [batchcommand]" << endl << endl <<

      "List of available commands:" << endl <<
      " cat <filename> [xrdcp parameters]" << endl <<
      "  outputs a file on standard output using xrdcp. <filename> can be a root:// URL." << endl <<
      " cd <dir name>" << endl <<
      "  changes the current directory" << endl <<
      "  Note: no existence check is performed." << endl <<
      " chmod <fileordirname> <user> <group> <other>" << endl <<
      "  modifies file permissions." << endl <<
      " connect [hostname[:port]]" << endl <<
      "  connects to the specified host." << endl <<
      " cp <fileordirname> <fileordirname> [xrdcp parameters]" << endl <<
      "  copies a file using xrdcp. <fileordirname> are always relative to the" << endl <<
      "  current remote path. Also, they can be root:// URLs specifying any other host." << endl <<
      " dirlist [dirname]" << endl <<
      "  gets the requested directory listing." << endl <<
      " dirlistrec [dirname]" << endl <<
      "  gets the requested recursive directory listing." << endl <<
      " envputint <varname> <intval>" << endl <<
      "  puts an integer in the internal environment." << endl <<
      " envputstring <varname> <stringval>" << endl <<
      "  puts a string in the internal environment." << endl <<
      " exit" << endl <<
      "  exits from the program." << endl <<
      " help" << endl <<
      "  this help screen." << endl <<
      " stat [fileordirname]" << endl <<
      "  gets info about the given file or directory path." << endl <<
      " statvfs [vfilesystempath]" << endl <<
      "  gets info about a virtual file system." << endl <<
      " existfile <filename>" << endl <<
      "  tells if the specified file exists." << endl <<
      " existdir <dirname>" << endl <<
      "  tells if the specified directory exists." << endl <<
      " getchecksum <filename>" << endl <<
      "  gets the checksum for the specified file." << endl <<
      " isfileonline <filename>" << endl <<
      "  tells if the specified file is online." << endl <<
      " locatesingle <filename> <writable>" << endl <<
      "  gives a location of the given file in the currently connected cluster." << endl <<
      "  if writable is true only a writable location is searched" << endl <<
      "   but, if no writable locations are found, the result is negative but may" << endl <<
      "   propose a non writable one." << endl <<
      " locateall <filename>" << endl <<
      "  gives all the locations of the given file in the currently connected cluster." << endl <<
      " mv <filename1> <filename2>" << endl <<
      "  moves filename1 to filename2 locally to the same server." << endl <<
      " mkdir <dirname> [user] [group] [other]" << endl <<
      "  creates a directory." << endl <<
      " rm <filename>" << endl <<
      "  removes a file." << endl <<
      " rmdir <dirname>" << endl <<
      "  removes a directory." << endl <<
      " prepare <filename> <options> <priority>" << endl <<
      "  stages a file in." << endl <<
      " query <reqcode> <parms>" << endl <<
      "  obtain server information" << endl <<
      " queryspace <logicalname>" << endl <<
      "  obtain space information" << endl <<
      endl <<
      "For further information, please read the xrootd protocol documentation." << endl <<
      endl;

   cout << endl;
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

void PrintLocateInfo(XrdClientLocate_Info &loc) {


	 cout << "InfoType: ";
	 switch (loc.Infotype) {
	 case XrdClientLocate_Info::kXrdcLocNone:
	   cout << "none" << endl;
	   break;
	 case XrdClientLocate_Info::kXrdcLocDataServer:
	   cout << "kXrdcLocDataServer" << endl;
	   break;
	 case XrdClientLocate_Info::kXrdcLocDataServerPending:
	   cout << "kXrdcLocDataServerPending" << endl;
	   break;
	 case XrdClientLocate_Info::kXrdcLocManager:
	   cout << "kXrdcLocManager" << endl;
	   break;
	 case XrdClientLocate_Info::kXrdcLocManagerPending:
	   cout << "kXrdcLocManagerPending" << endl;
	   break;
	 default:
	   cout << "Invalid Infotype" << endl;
	 }
	 cout << "CanWrite: ";
	 if (loc.CanWrite) cout << "true" << endl;
	 else cout << "false" << endl;
	 cout << "Location: '" << loc.Location << "'" << endl;
	 


}



// Main program
int main(int argc, char**argv) {

   int retval = 0;
   //signal(SIGINT, CtrlCHandler);

   DebugSetLevel(0);

   // We want this tool to be able to connect everywhere
   // Note that the side effect of these calls here is to initialize the
   // XrdClient environment.
   // This is crucial if we want to later override its default values
   EnvPutString( NAME_REDIRDOMAINALLOW_RE, "*" );
   EnvPutString( NAME_CONNECTDOMAINALLOW_RE, "*" );
   EnvPutString( NAME_REDIRDOMAINDENY_RE, "" );
   EnvPutString( NAME_CONNECTDOMAINDENY_RE, "" );

   EnvPutInt( NAME_DEBUG, -1);

   for (int i=1; i < argc; i++) {


      if ( (strstr(argv[i], "-O") == argv[i])) {
	 opaqueinfo=argv[i]+2;
	 continue;
      }

      if ( (strstr(argv[i], "-h") == argv[i]) || 
           (strstr(argv[i], "--help") == argv[i]) ) {
	 PrintUsage();
	 exit(0);
	 continue;
      }

      if ( (strstr(argv[i], "-DS") == argv[i]) &&
	   (argc >= i+2) ) {
	cerr << "Overriding " << argv[i]+3 << " with value " << argv[i+1] << ". ";
	 EnvPutString( argv[i]+3, argv[i+1] );
	 cerr << " Final value: " << EnvGetString(argv[i]+3) << endl;
	 i++;
	 continue;
      }

      if ( (strstr(argv[i], "-DI") == argv[i]) &&
	   (argc >= i+2) ) {
	cerr << "Overriding '" << argv[i]+3 << "' with value " << argv[i+1] << ". ";
	 EnvPutInt( argv[i]+3, atoi(argv[i+1]) );
	 cerr << " Final value: " << EnvGetLong(argv[i]+3) << endl;
	 i++;
	 continue;
      }

      // Any other par is ignored
      if ( (strstr(argv[i], "-") == argv[i]) && (strlen(argv[i]) > 1) ) {
	 cerr << "Unknown parameter " << argv[i] << endl;
	 continue;
      }

      if (!initialhost) initialhost = argv[i];
      else {
	cmdline_cmd += argv[i];
	cmdline_cmd += " ";
      }
   }

   DebugSetLevel(EnvGetLong(NAME_DEBUG));
   
   // if there's no command to execute from the cmdline...
   if (cmdline_cmd.length() == 0)
   cout << XRDCLI_VERSION << endl << "Welcome to the xrootd command line interface." << endl <<
      "Type 'help' for a list of available commands." << endl;



   if (initialhost) {
      XrdOucString s = "root://";
      s += initialhost;
      s += "//dummy";
      genadmin = new XrdClientAdmin(s.c_str());

      // Then connect
      if (!genadmin->Connect()) {
	 delete genadmin;
	 genadmin = 0;
      }
   }

   while(1) {
      stringstream prompt;
      char *linebuf=0;
      XrdOucTokenizer tkzer(linebuf); // should add valid XrdOucTokenizer constructor()

      if (cmdline_cmd.length() == 0) {
	PrintPrompt(prompt);
	linebuf = readline(prompt.str().c_str());
	if(! linebuf || ! *linebuf) {
	  free(linebuf);
	  continue;
	}
#ifdef HAVE_READLINE
	add_history(linebuf);
#endif
      }
      else linebuf = strdup(cmdline_cmd.c_str());

      // And the simple parsing starts
      tkzer.Attach(linebuf);
      if (!tkzer.GetLine()) continue;
      
      char *cmd = tkzer.GetToken(0, 1);
      
      if (!cmd) continue;
      
      // -------------------------- cd ---------------------------
      if (!strcmp(cmd, "cd")) {
	 char *parmname = tkzer.GetToken(0, 0);

	 if (!parmname || !strlen(parmname)) {
	    cout << "A directory name is needed." << endl << endl;
	    retval = 1;
	 }

	 // Quite trivial directory processing
	 if (!strcmp(parmname, "..")) {
            if (currentpath == "/") continue;

	    int pos = currentpath.rfind('/');

	    if (pos != STR_NPOS)
	       currentpath.erase(pos);

            if (!currentpath.length()) {
               currentpath = "/";
            }

	    retval = 1;
            continue;
	 }
         else
            if (!strcmp(parmname, ".")) {
               retval = 1;
               continue;
            }
	    
         if (!currentpath.length() || (currentpath[currentpath.length()-1] != '/')) {
            currentpath += "/";
         }

         if (parmname[0] == '/') currentpath = parmname;
         else
            currentpath += parmname;

      }
      else

      // -------------------------- envputint ---------------------------
      if (!strcmp(cmd, "envputint")) {
	 char *parmname = tkzer.GetToken(0, 0),
	    *val = tkzer.GetToken(0, 1);

	 if (!parmname || !val) {
	    cout << "A parameter name and an integer value are needed." << endl << endl;
	    retval = 1;
	 } else {

            EnvPutInt(parmname, atoi(val));
            DebugSetLevel(EnvGetLong(NAME_DEBUG));
         }

      }
      else
      // -------------------------- envputstring ---------------------------
      if (!strcmp(cmd, "envputstring")) {
	 char *parmname = tkzer.GetToken(0, 0),
	    *val = tkzer.GetToken(0, 1);

	 if (!parmname || !val) {
	    cout << "A parameter name and a string value are needed." << endl << endl;
	    retval = 1;
	 } else {

            EnvPutString(parmname, val);

         }

      }
      else
      // -------------------------- help ---------------------------
      if (!strcmp(cmd, "help")) {
	 PrintHelp();

      }
      else
      // -------------------------- exit ---------------------------
      if (!strcmp(cmd, "exit")) {
	 cout << "Goodbye." << endl << endl;
	 retval = 0;
	 break;
      }
      else
      // -------------------------- connect ---------------------------
      if (!strcmp(cmd, "connect")) {
	 char *host = initialhost;

	 // If no host was given, then pretend one
	 if (!host) {

	    host = tkzer.GetToken(0, 1);
	    if (!host || !strlen(host)) {
	       cout << "A hostname is needed to connect somewhere." << endl;
	       retval = 1;
	    }

	 }

         if (!retval) {
            // Init the instance
            if (genadmin) delete genadmin;
            XrdOucString h(host);
            h  = "root://" + h;
            h += "//dummy";

            genadmin = new XrdClientAdmin(h.c_str());

            // Then connect
            if (!genadmin->Connect()) {
               delete genadmin;
               genadmin = 0;
            }
         }

      }
      else
      // -------------------------- dirlistrec ---------------------------
         if (!strcmp(cmd, "dirlistrec")) {
            XrdClientVector<XrdOucString> pathq;

            if (!genadmin) {
               cout << "Not connected to any server." << endl;
               retval = 1;
            }

            if (!retval) {

               char *dirname = tkzer.GetToken(0, 0);
               XrdOucString path;

               if (dirname) {
                  if (dirname[0] == '/')
                     path = dirname;
                  else {
                     if ((currentpath.length() > 0) && (currentpath[currentpath.length()-1] != '/'))
                        path = currentpath + "/" + dirname;
                     else
                        path = currentpath + dirname;

                  }
               }
               else path = currentpath;

               if (!path.length()) {
                  cout << "The current path is an empty string. Assuming '/'." << endl;
                  path = '/';
               }


               // Initialize the queue with this path
               pathq.Push_back(path);


               while (pathq.GetSize() > 0) {
                  XrdOucString pathtodo = pathq.Pop_back();

                  // Now try to issue the request
                  XrdClientVector<XrdClientAdmin::DirListInfo> nfo;
                  if (!genadmin->DirList(pathtodo.c_str(), nfo, true)) {
                     retval = 1;  
                     cout << "Error listing path '" << pathtodo << "' in server " <<
                        genadmin->GetCurrentUrl().HostWPort <<
                        " The path does not exist in some servers or there are malformed filenames." << endl;
                  }

                  // Now check the answer
                  if (!CheckAnswer(genadmin)) {
                     retval = 1;
                     cout << "Error '" << genadmin->LastServerError()->errmsg <<
                        "' listing path '" << pathtodo <<
                        "' in server" << genadmin->GetCurrentUrl().HostWPort <<
                        " or in some of its child nodes." << endl;
                  }
      
                  for (int i = 0; i < nfo.GetSize(); i++) {

                     if ((nfo[i].flags & kXR_isDir) &&
                         (nfo[i].flags & kXR_readable) &&
                         (nfo[i].flags & kXR_xset)) {
                  

                        // The path has not to be pushed if it's already present
                        // This may happen if several servers have the same path
                        bool foundpath = false;
                        for (int ii = 0; ii < pathq.GetSize(); ii++) {
                           if (nfo[i].fullpath == pathq[ii]) {
                              foundpath = true;
                              break;
                           }
                        }
                  
                        if (!foundpath)
                           pathq.Push_back(nfo[i].fullpath);
                        else 
                           // If the path is already present in the queue then it was already printed as well.
                           continue;

                     }

                     char ts[256];
                     strcpy(ts, "n/a");

                     struct tm *t = gmtime(&nfo[i].modtime);
                     strftime(ts, 255, "%F %T", t);

                     char cflgs[16];
                     memset(cflgs, 0, 16);

                     if (nfo[i].flags & kXR_isDir)
                        strcat(cflgs, "d");
                     else strcat(cflgs, "-");

                     if (nfo[i].flags & kXR_readable)
                        strcat(cflgs, "r");
                     else strcat(cflgs, "-");

                     if (nfo[i].flags & kXR_writable)
                        strcat(cflgs, "w");
                     else strcat(cflgs, "-");

                     if (nfo[i].flags & kXR_xset)
                        strcat(cflgs, "x");
                     else strcat(cflgs, "-");

                     printf("%s(%03ld) %12lld %s %s\n", cflgs, nfo[i].flags, nfo[i].size, ts, nfo[i].fullpath.c_str());

                  }
            
                  if (nfo.GetSize()) cout << endl;
               }

               if (retval) cout << "Errors during processing. Please check them." << endl;
            }
      }
      else
      // -------------------------- dirlist ---------------------------
      if (!strcmp(cmd, "dirlist")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {


            char *dirname = tkzer.GetToken(0, 0);
            XrdOucString path;

            if (dirname) {
               if (dirname[0] == '/')
                  path = dirname;
               else {
                  if ((currentpath.length() > 0) && (currentpath[currentpath.length()-1] != '/'))
                     path = currentpath + "/" + dirname;
                  else
                     path = currentpath + dirname;

               }
            }
            else path = currentpath;

            if (!path.length()) {
               cout << "The current path is an empty string. Assuming '/'." << endl;
               path = '/';
            }

            // Now try to issue the request
            XrdClientVector<XrdClientAdmin::DirListInfo> nfo;
            if (!genadmin->DirList(path.c_str(), nfo, true)) {
               //nfo.Clear();
               retval = 1;
            
            }

            // Now check the answer
            if (!CheckAnswer(genadmin)) {
               retval = 1;
               //nfo.Clear();
            }
      

            for (int i = 0; i < nfo.GetSize(); i++) {
               char ts[256];
               strcpy(ts, "n/a");

               struct tm *t = gmtime(&nfo[i].modtime);
               strftime(ts, 255, "%F %T", t);

               char cflgs[16];
               memset(cflgs, 0, 16);

               if (nfo[i].flags & kXR_isDir)
                  strcat(cflgs, "d");
               else strcat(cflgs, "-");

               if (nfo[i].flags & kXR_readable)
                  strcat(cflgs, "r");
               else strcat(cflgs, "-");

               if (nfo[i].flags & kXR_writable)
                  strcat(cflgs, "w");
               else strcat(cflgs, "-");

               if (nfo[i].flags & kXR_xset)
                  strcat(cflgs, "x");
               else strcat(cflgs, "-");

               printf("%s(%03ld) %12lld %s %s\n", cflgs, nfo[i].flags, nfo[i].size, ts, nfo[i].fullpath.c_str());

            }

            if (retval) cout << "Errors during processing. Please check." << endl;
            cout << endl;
         }

      }
      else
      // -------------------------- locatesingle ---------------------------
      if (!strcmp(cmd, "locatesingle")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {
            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else pathname = currentpath;

            char *writable = tkzer.GetToken(0, 1);
            bool wrt = false;

            if (writable) {
               wrt = true;
               if (!strcmp(writable, "false") ||
                   !strcmp(writable, "0")) wrt = false;
               else
                  cout << "Checking for a writable location." << endl;
            }

            // Now try to issue the request
            XrdClientLocate_Info loc;
            bool r;
            r = genadmin->Locate((kXR_char *)pathname.c_str(), loc, wrt);
            if (!r)
               cout << "No matching files were found." << endl;
	   
            PrintLocateInfo(loc);



            cout << endl;
         }


 
      }
      else
      // -------------------------- locateall ---------------------------
      if (!strcmp(cmd, "locateall")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else pathname = currentpath;

            // Now try to issue the request
            XrdClientVector<XrdClientLocate_Info> loc;
            bool r;
            r = genadmin->Locate((kXR_char *)pathname.c_str(), loc);
            if (!r)
               cout << "No matching files were found." << endl;
	 
            for (int ii = 0; ii < loc.GetSize(); ii++) {
               cout << endl << endl << "------------- Location #" << ii+1 << endl;
               PrintLocateInfo(loc[ii]);
            }

            cout << endl;

         }

      }
      else
      // -------------------------- stat ---------------------------
      if (!strcmp(cmd, "stat")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }


         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else pathname = currentpath;

            // Now try to issue the request
            long id, flags, modtime;
            long long size;
            genadmin->Stat(pathname.c_str(), id, size, flags, modtime);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;
      
            cout << "Id: " << id << " Size: " << size << " Flags: " << flags << " Modtime: " << modtime << endl;

            cout << endl;
         }

      }
      else
      // -------------------------- statvfs ---------------------------
      if (!strcmp(cmd, "statvfs")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else pathname = currentpath;

            // Now try to issue the request
            int rwservers = 0;
            long long rwfree = 0;
            int rwutil = 0;
            int stagingservers = 0;
            long long stagingfree = 0;
            int stagingutil = 0;


            genadmin->Stat_vfs(pathname.c_str(), rwservers, rwfree, rwutil,
                               stagingservers, stagingfree, stagingutil);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;
      
            cout << "r/w nodes: " << rwservers <<
               " r/w free space: " << rwfree <<
               " r/w utilization: " << rwutil <<
               " staging nodes: " << rwservers <<
               " staging free space: " << rwfree <<
               " staging utilization: " << rwutil << endl;

            cout << endl;

         }

      }
      else
      // -------------------------- ExistFile ---------------------------
      if (!strcmp(cmd, "existfile")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }


         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else pathname = currentpath;

            // Now try to issue the request
            vecBool vb;
            vecString vs;
            vs.Push_back(pathname);
            genadmin->ExistFiles(vs, vb);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;
      
            if (vb[0])
               cout << "The file exists." << endl;
            else
               cout << "File not found." << endl;

            cout << endl;

         }
	 
      }
      else
      // -------------------------- ExistDir ---------------------------
      if (!strcmp(cmd, "existdir")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }


         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else pathname = currentpath;

            // Now try to issue the request
            vecBool vb;
            vecString vs;
            vs.Push_back(pathname);
            genadmin->ExistDirs(vs, vb);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;
      
            if (vb[0])
               cout << "The directory exists." << endl;
            else
               cout << "Directory not found." << endl;

            cout << endl;
         }

	 
      }
      else

      // -------------------------- getchecksum ---------------------------
      if (!strcmp(cmd, "getchecksum")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }


         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            };

            // Now try to issue the request
            kXR_char *ans;

            genadmin->GetChecksum((kXR_char *)pathname.c_str(), &ans);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;

            cout << "Checksum: " << ans << endl;
            cout << endl;

            free(ans);
         }

      }
      else
      // -------------------------- isfileonline ---------------------------
      if (!strcmp(cmd, "isfileonline")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }

            // Now try to issue the request
            vecBool vb;
            vecString vs;
            vs.Push_back(pathname);
            genadmin->IsFileOnline(vs, vb);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;
      
            if (vb[0])
               cout << "The file is online." << endl;
            else
               cout << "The file is not online." << endl;

            cout << endl;

         }

      }
      else
      // -------------------------- mv ---------------------------
      if (!strcmp(cmd, "mv")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {
            char *fname1 = tkzer.GetToken(0, 0);
            char *fname2 = tkzer.GetToken(0, 0);
            XrdOucString pathname1, pathname2;

            if (fname1) {
               if (fname1[0] == '/')
                  pathname1 = fname1;
               else
                  pathname1 = currentpath + "/" + fname1;
            }
            else {
               cout << "Two parameters are mandatory." << endl;
               retval = 1;
            }
            if (fname2) {
               if (fname2[0] == '/')
                  pathname2 = fname2;
               else
                  pathname2 = currentpath + "/" + fname2;
            }
            else {
               cout << "Two parameters are mandatory." << endl;
               retval = 1;
            }

            // Now try to issue the request
            genadmin->Mv(pathname1.c_str(), pathname2.c_str());

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;

            cout << endl;
         }
	
      }
      else
      // -------------------------- mkdir ---------------------------
      if (!strcmp(cmd, "mkdir")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {
            char *fname1 = tkzer.GetToken(0, 0);
            char *userc = tkzer.GetToken(0, 0);
            char *groupc = tkzer.GetToken(0, 0);
            char *otherc = tkzer.GetToken(0, 0);

            int user = 0, group = 0, other = 0;
            if (userc) user = atoi(userc);
            if (groupc) group = atoi(groupc);
            if (otherc) other = atoi(otherc);

            XrdOucString pathname1;

            if (fname1) {
               if (fname1[0] == '/')
                  pathname1 = fname1;
               else
                  pathname1 = currentpath + "/" + fname1;
            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }


            // Now try to issue the request
            genadmin->Mkdir(pathname1.c_str(), user, group, other);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;

            cout << endl;
         }
	 
      }
      else
      // -------------------------- chmod ---------------------------
      if (!strcmp(cmd, "chmod")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname1 = tkzer.GetToken(0, 0);
            char *userc = tkzer.GetToken(0, 0);
            char *groupc = tkzer.GetToken(0, 0);
            char *otherc = tkzer.GetToken(0, 0);

            int user = 0, group = 0, other = 0;
            if (userc) user = atoi(userc);
            if (groupc) group = atoi(groupc);
            if (otherc) other = atoi(otherc);

            XrdOucString pathname1;

            if (fname1) {
               if (fname1[0] == '/')
                  pathname1 = fname1;
               else
                  pathname1 = currentpath + "/" + fname1;
            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }


            // Now try to issue the request
            genadmin->Chmod(pathname1.c_str(), user, group, other);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;

            cout << endl;
         }
      }
      else
      // -------------------------- truncate ---------------------------
      if (!strcmp(cmd, "truncate")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            char *slen = tkzer.GetToken(0, 0);

            long long len = 0;
            if (slen) len = atoll(slen);
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }

            if (len <= 0) {
               cout << "Bad length." << endl;
               retval = 1;
            }

            XrdOucString pathname1;

            if (fname) {
               if (fname[0] == '/')
                  pathname1 = fname;
               else
                  pathname1 = currentpath + "/" + fname;
            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }


            // Now try to issue the request
            genadmin->Truncate(pathname1.c_str(), len);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;

            cout << endl;

         }


      }
      else
      // -------------------------- rm ---------------------------
      if (!strcmp(cmd, "rm")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }

            // Now try to issue the request
            genadmin->Rm(pathname.c_str());

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;

            cout << endl;
         } 
      }
      else
      // -------------------------- rmdir ---------------------------
      if (!strcmp(cmd, "rmdir")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname = tkzer.GetToken(0, 0);
            XrdOucString pathname;

            if (fname) {
               if (fname[0] == '/')
                  pathname = fname;
               else
                  pathname = currentpath + "/" + fname;
            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }

            // Now try to issue the request
            genadmin->Rmdir(pathname.c_str());

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;

            cout << endl;
         }

      }
      else
      // -------------------------- prepare ---------------------------
      if (!strcmp(cmd, "prepare")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname1 = tkzer.GetToken(0, 0);
            char *optsc = tkzer.GetToken(0, 0);
            char *prioc = tkzer.GetToken(0, 0);

            int opts = 0, prio = 0;
            if (optsc) opts = atoi(optsc);
            if (prioc) prio = atoi(prioc);

            XrdOucString pathname1;

            if (fname1) {
               if (fname1[0] == '/')
                  pathname1 = fname1;
               else
                  pathname1 = currentpath + "/" + fname1;
            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }

            // Now try to issue the request
            vecString vs;
            vs.Push_back(pathname1);
            genadmin->Prepare(vs, (kXR_char)opts, (kXR_char)prio);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;

            cout << endl;

         }


      }
      else
      // -------------------------- cat ---------------------------
      if (!strcmp(cmd, "cat")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname1 = tkzer.GetToken(0, 0);
            char *tk;
            XrdOucString pars;

            while ((tk = tkzer.GetToken(0, 0))) {
               pars += " ";
               pars += tk;
            }

            XrdOucString pathname1;

            if (fname1) {

               if ( (strstr(fname1, "root://") == fname1) ||
                    (strstr(fname1, "xroot://") == fname1) )
                  pathname1 = fname1;
               else
                  if (fname1[0] == '/') {
                     pathname1 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                     pathname1 += "/";
                     pathname1 += fname1;
                  }
                  else {
                     pathname1 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                     pathname1 += "/";
                     pathname1 += currentpath;
                     pathname1 += "/";
                     pathname1 += fname1;
                  }

            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }

            XrdOucString cmd;

            cmd = "xrdcp -s ";
            cmd += pathname1;
            cmd += pars;
            cmd += " -";

            int rt = system(cmd.c_str());

            cout << "cat returned " << rt << endl;

            cout << endl;
         }

	
      }
      else
      // -------------------------- cp ---------------------------
      if (!strcmp(cmd, "cp")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *fname1 = tkzer.GetToken(0, 0);
            char *fname2 = tkzer.GetToken(0, 0);
            char *tk;
            XrdOucString pars;

            while ((tk = tkzer.GetToken(0, 0))) {
               pars += " ";
               pars += tk;
            }

            XrdOucString pathname1, pathname2;

            if (fname1) {

               if ( (strstr(fname1, "root://") == fname1) ||
                    (strstr(fname1, "xroot://") == fname1) )
                  pathname1 = fname1;
               else
                  if (fname1[0] == '/') {
                     pathname1 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                     pathname1 += "/";
                     pathname1 += fname1;
                  }
                  else {
                     pathname1 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                     pathname1 += "/";
                     pathname1 += currentpath;
                     pathname1 += "/";
                     pathname1 += fname1;
                  }

            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }
            if (fname2) {

               if ( (strstr(fname2, "root://") == fname2) ||
                    (strstr(fname2, "xroot://") == fname2) )
                  pathname2 = fname2;
               else
                  if (fname2[0] == '/') {
                     pathname2 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                     pathname2 += "/";
                     pathname2 += fname2;
                  }
                  else {
                     pathname2 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                     pathname2 += "/";
                     pathname2 += currentpath;
                     pathname2 += "/";
                     pathname2 += fname2;
                  }

            }
            else {
               cout << "Missing parameter." << endl;
               retval = 1;
            }

            XrdOucString cmd;

            cmd = "xrdcp ";
            cmd += pathname1;
            cmd += " ";
            cmd += pathname2 + pars;

            int rt = system(cmd.c_str());

            cout << "cp returned " << rt << endl;

            cout << endl;
         }


      }
      else
      // -------------------------- query ---------------------------
      if (!strcmp(cmd, "query")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *reqcode = tkzer.GetToken(0, 0);
            const kXR_char *args = (const kXR_char *)tkzer.GetToken(0, 0);
            kXR_char Resp[1024];

            genadmin->Query(atoi(reqcode), args, Resp, 1024);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;
      
            cout << Resp << endl;

            cout << endl;
         }
	
      }
      else
      // -------------------------- queryspace ---------------------------
      if (!strcmp(cmd, "queryspace")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    retval = 1;
	 }

         if (!retval) {

            char *ns = tkzer.GetToken(0, 0);
            long long totspace;
            long long totfree;
            long long totused;
            long long largestchunk;

            genadmin->GetSpaceInfo(ns, totspace, totfree, totused, largestchunk);

            // Now check the answer
            if (!CheckAnswer(genadmin))
               retval = 1;
      
            cout << "Disk space approximations (MB):" << endl <<
               "Total         : " << totspace/(1024*1024) << endl <<
               "Free          : " << totfree/(1024*1024) << endl <<
               "Used          : " << totused/(1024*1024) << endl <<
               "Largest chunk : " << largestchunk/(1024*1024) << endl;

            cout << endl;
         }



      }
      else {

	// ---------------------------------------------------------------------
	// -------------------------- not recognized ---------------------------
	cout << "Command not recognized." << endl <<
	  "Type \"help\" for a list of commands." << endl << endl;
      }

      
      delete[] linebuf;

      // if it was a cmd from the commandline...
      if (cmdline_cmd.length() > 0) break;

   } // while (1)


   if (genadmin) delete genadmin;
   return retval;

}
