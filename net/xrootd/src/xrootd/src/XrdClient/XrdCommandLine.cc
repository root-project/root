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

#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdOuc/XrdOucTokenizer.hh"

#include <stdio.h>
#include <iostream>
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





#define XRDCLI_VERSION            "(C) 2004 SLAC INFN xrd 0.1 beta"


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
XrdOucString currentpath;

///////////////////////

void PrintUsage() {
   cerr << "usage: xrd [host] "
     "[-DSparmname stringvalue] ... [-DIparmname intvalue]  [-O<opaque info>]" << endl;
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
      "List of available commands:" << endl <<
      " cd <dir name>" << endl <<
      "  changes the current directory" << endl <<
      "  Note: no existence check is performed." << endl <<
      " envputint <varname> <intval>" << endl <<
      "  puts an integer in the internal environment." << endl <<
      " envputstring <varname> <stringval>" << endl <<
      "  puts a string in the internal environment." << endl <<
      " help" << endl <<
      "  this help screen." << endl <<
      " exit" << endl <<
      "  exits from the program." << endl <<
      " connect [hostname[:port]]" << endl <<
      "  connects to the specified host." << endl <<
      " dirlist [dirname]" << endl <<
      "  gets the requested directory listing." << endl <<
      " stat [fileordirname]" << endl <<
      "  gets info about the given file or directory path." << endl <<
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
      " chmod <fileordirname> <user> <group> <other>" << endl <<
      "  modifies file permissions." << endl <<
      " rm <filename>" << endl <<
      "  removes a file." << endl <<
      " rmdir <dirname>" << endl <<
      "  removes a directory." << endl <<
      " prepare <filename> <options> <priority>" << endl <<
      "  stages a file in." << endl <<
      " cat <filename> [xrdcp parameters]" << endl <<
      "  outputs a file on standard output using xrdcp. <filename> can be a root:// URL." << endl <<
      " cp <fileordirname> <fileordirname> [xrdcp parameters]" << endl <<
      "  copies a file using xrdcp. <fileordirname> are always relative to the" << endl <<
      "  current remote path. Also, they can be root:// URLs specifying any other host." << endl <<
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

      if ( (strstr(argv[i], "-h") == argv[i])) {
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

   }

   DebugSetLevel(EnvGetLong(NAME_DEBUG));

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

      
      PrintPrompt(prompt);
      linebuf = readline(prompt.str().c_str());
      if(! linebuf || ! *linebuf) {
        free(linebuf);
	continue;
      }
#ifdef HAVE_READLINE
      add_history(linebuf);
#endif

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
	    continue;
	 }

	 // Quite trivial directory processing
	 if (!strcmp(parmname, "..")) {
	    int pos = currentpath.rfind('/');

	    if (pos != STR_NPOS)
	       currentpath.erase(pos);

	    continue;
	 }

	 if (!strcmp(parmname, "."))
	    continue;
	    
	 currentpath += "/";
	 currentpath += parmname;
	 continue;
      }


      // -------------------------- envputint ---------------------------
      if (!strcmp(cmd, "envputint")) {
	 char *parmname = tkzer.GetToken(0, 0),
	    *val = tkzer.GetToken(0, 1);

	 if (!parmname || !val) {
	    cout << "A parameter name and an integer value are needed." << endl << endl;
	    continue;
	 }

	 EnvPutInt(parmname, atoi(val));
	 DebugSetLevel(EnvGetLong(NAME_DEBUG));
	 continue;
      }

      // -------------------------- envputstring ---------------------------
      if (!strcmp(cmd, "envputstring")) {
	 char *parmname = tkzer.GetToken(0, 0),
	    *val = tkzer.GetToken(0, 1);

	 if (!parmname || !val) {
	    cout << "A parameter name and a string value are needed." << endl << endl;
	    continue;
	 }

	 EnvPutString(parmname, val);
	 continue;
      }

      // -------------------------- help ---------------------------
      if (!strcmp(cmd, "help")) {
	 PrintHelp();
	 continue;
      }

      // -------------------------- exit ---------------------------
      if (!strcmp(cmd, "exit")) {
	 cout << "Goodbye." << endl << endl;
	 retval = 0;
	 break;
      }

      // -------------------------- connect ---------------------------
      if (!strcmp(cmd, "connect")) {
	 char *host = initialhost;

	 // If no host was given, then pretend one
	 if (!host) {

	    host = tkzer.GetToken(0, 1);
	    if (!host || !strlen(host)) {
	       cout << "A hostname is needed to connect somewhere." << endl;
	       continue;
	    }

	 }

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

	 continue;
      }

      // -------------------------- dirlist ---------------------------
      if (!strcmp(cmd, "dirlist")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

	 char *dirname = tkzer.GetToken(0, 0);
	 XrdOucString path;

	 if (dirname) {
	    if (dirname[0] == '/')
	       path = dirname;
	    else
	       path = currentpath + "/" + dirname;
	 }
	 else path = currentpath;

	 // Now try to issue the request
	 vecString vs;
	 genadmin->DirList(path.c_str(), vs);

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;
      
	 for (int i = 0; i < vs.GetSize(); i++)
	    cout << vs[i] << endl;

	 cout << endl;
	 continue;
      }

      // -------------------------- locatesingle ---------------------------
      if (!strcmp(cmd, "locatesingle")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	 continue;
      }


      // -------------------------- locateall ---------------------------
      if (!strcmp(cmd, "locateall")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	 continue;
      }

      // -------------------------- stat ---------------------------
      if (!strcmp(cmd, "stat")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
      
	 cout << "Id: " << id << " Size: " << size << " Flags: " << flags << " Modtime: " << modtime << endl;

	 cout << endl;
	 continue;
      }

      // -------------------------- ExistFile ---------------------------
      if (!strcmp(cmd, "existfile")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
      
	 if (vb[0])
	    cout << "The file exists." << endl;
	 else
	    cout << "File not found." << endl;

	 cout << endl;
	 continue;
      }

      // -------------------------- ExistDir ---------------------------
      if (!strcmp(cmd, "existdir")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
      
	 if (vb[0])
	    cout << "The directory exists." << endl;
	 else
	    cout << "Directory not found." << endl;

	 cout << endl;
	 continue;
      }


      // -------------------------- getchecksum ---------------------------
      if (!strcmp(cmd, "getchecksum")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 };

	 // Now try to issue the request
	 kXR_char *ans;

	 genadmin->GetChecksum((kXR_char *)pathname.c_str(), &ans);

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;

	 cout << "Checksum: " << ans << endl;
	 cout << endl;

	 free(ans);

	 continue;
      }

      // -------------------------- isfileonline ---------------------------
      if (!strcmp(cmd, "isfileonline")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 }

	 // Now try to issue the request
	 vecBool vb;
	 vecString vs;
	 vs.Push_back(pathname);
	 genadmin->IsFileOnline(vs, vb);

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;
      
	 if (vb[0])
	    cout << "The file is online." << endl;
	 else
	    cout << "The file is not online." << endl;

	 cout << endl;
	 continue;
      }

      // -------------------------- mv ---------------------------
      if (!strcmp(cmd, "mv")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 }
	 if (fname2) {
	    if (fname2[0] == '/')
	       pathname2 = fname2;
	    else
	       pathname2 = currentpath + "/" + fname2;
	 }
	 else {
	    cout << "Two parameters are mandatory." << endl;
	    continue;
	 }

	 // Now try to issue the request
	 genadmin->Mv(pathname1.c_str(), pathname2.c_str());

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;

	 cout << endl;
	 continue;
      }

      // -------------------------- mkdir ---------------------------
      if (!strcmp(cmd, "mkdir")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 }


	 // Now try to issue the request
	 genadmin->Mkdir(pathname1.c_str(), user, group, other);

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;

	 cout << endl;
	 continue;
      }

      // -------------------------- chmod ---------------------------
      if (!strcmp(cmd, "chmod")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 }


	 // Now try to issue the request
	 genadmin->Chmod(pathname1.c_str(), user, group, other);

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;

	 cout << endl;
	 continue;
      }


      // -------------------------- rm ---------------------------
      if (!strcmp(cmd, "rm")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 }

	 // Now try to issue the request
	 genadmin->Rm(pathname.c_str());

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;

	 cout << endl;
	 continue;
      }

      // -------------------------- rmdir ---------------------------
      if (!strcmp(cmd, "rmdir")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 }

	 // Now try to issue the request
	 genadmin->Rmdir(pathname.c_str());

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;

	 cout << endl;
	 continue;
      }


      // -------------------------- prepare ---------------------------
      if (!strcmp(cmd, "prepare")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 }


	 // Now try to issue the request
	 vecString vs;
	 vs.Push_back(pathname1);
	 genadmin->Prepare(vs, (kXR_char)opts, (kXR_char)prio);

	 // Now check the answer
	 if (!CheckAnswer(genadmin))
	    continue;

	 cout << endl;
	 continue;
      }

      // -------------------------- cat ---------------------------
      if (!strcmp(cmd, "cat")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
	 }

	 XrdOucString cmd;

	 cmd = "xrdcp -s ";
	 cmd += pathname1;
	 cmd += pars;
	 cmd += " -";

	 int rt = system(cmd.c_str());

	 cout << "cat returned " << rt << endl;

	 cout << endl;
	 continue;
      }

      // -------------------------- cp ---------------------------
      if (!strcmp(cmd, "cp")) {

	 if (!genadmin) {
	    cout << "Not connected to any server." << endl;
	    continue;
	 }

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
	    continue;
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
	    continue;
	 }

	 XrdOucString cmd;

	 cmd = "xrdcp ";
	 cmd += pathname1;
	 cmd += " ";
	 cmd += pathname2 + pars;

	 int rt = system(cmd.c_str());

	 cout << "cp returned " << rt << endl;

	 cout << endl;
	 continue;
      }


      // ---------------------------------------------------------------------
      // -------------------------- not recognized ---------------------------
      cout << "Command not recognized." << endl <<
	 "Type \"help\" for a list of commands." << endl << endl;
      

      
      free(linebuf);
   } // while (1)


   if (genadmin) delete genadmin;
   return retval;

}
