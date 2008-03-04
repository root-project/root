/////////////////////////////////////////////////////////////////////////
//                                                                      //
// xrdmn                                                                //
//                                                                      //
// Author: Andreas Joachim Peters (CERN,2005)                           //
// A rm-like command line tool for xrootd environments                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id: xrm.cc

#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdarg.h>

extern "C" {
/////////////////////////////////////////////////////////////////////
// function + macro to allow formatted print via cout,cerr
/////////////////////////////////////////////////////////////////////
void cout_print(const char *format, ...)
{
  char cout_buff[4096];
  va_list args;
  va_start(args, format);
  vsprintf(cout_buff, format,  args);
  va_end(args);
  cout << cout_buff;
}

void cerr_print(const char *format, ...)
{
  char cerr_buff[4096];
  va_list args;
  va_start(args, format);
  vsprintf(cerr_buff, format,  args);
  va_end(args);
  cerr << cerr_buff;
}

#define COUT(s) do { \
  cout_print s;      \
} while (0)

#define CERR(s) do { \
  cerr_print s;      \
} while (0)

}
//////////////////////////////////////////////////////////////////////


#define XRDRM_VERSION            "(C) 2004 SLAC INFN xrdrm 0.1"

bool summary=false;
char* authz=0;
char authzfilename[4096]="";
struct timeval abs_start_time;
struct timeval abs_stop_time;
struct timezone tz;


void print_summary(const char* del) {
  gettimeofday (&abs_stop_time, &tz);
  float abs_time=((float)((abs_stop_time.tv_sec - abs_start_time.tv_sec) *1000 +
			      (abs_stop_time.tv_usec - abs_start_time.tv_usec) / 1000));

  XrdOucString xdel(del);

  xdel.erase(xdel.rfind('?'));
  
  COUT(("[xrootd] #################################################################\n"));
  COUT(("[xrootd] # Deletion Name            : %s\n",xdel.c_str()));
  COUT(("[xrootd] # Realtime [s]             : %f\n",abs_time/1000.0));
  COUT(("[xrootd] ##########################################################\n"));
}

void PrintUsage() {
   cerr << "usage: xrm <file> [file2 [file3 [...] ]]"
     "[-DSparmname stringvalue] ... [-DIparmname intvalue] [-s] [-ns] [-v] [-authz <authz-file>] [-force]" << endl;
   cerr << " -s   :         silent mode, no summary output" << endl;
   cerr << " -v   :         display summary output" << endl <<endl;
   cerr << " -authz <authz-file> : set the authorization file" << endl;
   cerr << " -force              : set 'eternal'(1week) connect timeouts to block until xcp is executed" << endl << endl;
   cerr << " where:" << endl;
   cerr << "   parmname     is the name of an internal parameter" << endl;
   cerr << "   stringvalue  is a string to be assigned to an internal parameter" << endl;
   cerr << "   intvalue     is an int to be assigned to an internal parameter" << endl;
}

// Main program
int main(int argc, char**argv) {
  char *delpath = 0;
  int retval = -1;
  unsigned int rmstartarg=0;
   if (argc < 2) {
      PrintUsage();
      exit(1);
   }

   DebugSetLevel(20000);

   // We want this tool to be able to copy from/to everywhere
   // Note that the side effect of these calls here is to initialize the
   // XrdClient environment.
   // This is crucial if we want to later override its default values
   EnvPutString( NAME_REDIRDOMAINALLOW_RE, "*" );
   EnvPutString( NAME_CONNECTDOMAINALLOW_RE, "*" );
   EnvPutString( NAME_REDIRDOMAINDENY_RE, "" );
   EnvPutString( NAME_CONNECTDOMAINDENY_RE, "" );
   EnvPutInt( NAME_CONNECTTIMEOUT , 2);
   EnvPutInt( NAME_RECONNECTTIMEOUT , 2);
   EnvPutInt( NAME_FIRSTCONNECTMAXCNT, 1);
   EnvPutInt( NAME_DEBUG, -1);

   for (int i=1; i < argc; i++) {
     
      if ( (strstr(argv[i], "-v") == argv[i])) {
	summary=true;
	continue;
      }

      if ( (strstr(argv[i], "-authz") == argv[i]) &&
	   (argc >= i+2) ) {
	strcpy(authzfilename,argv[i+1]);
	 i++;
 	 continue;
      }

      if ( (strstr(argv[i], "-force") == argv[i])) {
 	 EnvPutInt( NAME_CONNECTTIMEOUT , 60);
	 EnvPutInt( NAME_FIRSTCONNECTMAXCNT, 7*24*60);
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

      rmstartarg = i;
      break;
   }
   

   // load the authz structure from an environment variable or the given authz file
   authz = getenv("IO_AUTHZ");
   if (strlen(authzfilename)) {
     int fdauthz = open(authzfilename,O_RDONLY);
     if (!fdauthz) {
       cerr << "Error xcp: cannot open authz file %s " << authzfilename << endl;
       exit(-1);
     }
     struct stat stat;
     if (fstat(fdauthz, &stat)) {
       cerr << "Error xcp: cannot stat authz file %s " << authzfilename << endl;
       exit(-1);
     }
     // set 16M limit for authz files
     if (stat.st_size < 1024*1024*16) {
       authz = (char*) malloc (1024*1024*16);
       if (!authz) {
	 cerr << "Error xcp: cannot allocate authz memory " << endl;
	 exit(-1);
       }
       int nread = read(fdauthz,authz,stat.st_size);
       if (nread!=stat.st_size) {
	 cerr << "Error xcp: error reading authz file %s " << authzfilename << endl;
	 exit(-1);
       }
       // terminate the string
       authz[stat.st_size] = 0;
     } else {
       cerr << "Error xcp: authz file %s exceeds the 16M limit " << authzfilename << endl;
       exit(-1);
     }
     close(fdauthz);
   }

   DebugSetLevel(EnvGetLong(NAME_DEBUG));

   Info(XrdClientDebug::kNODEBUG, "main", XRDRM_VERSION);
    

   for (int loop = rmstartarg; loop < argc; loop++) {
     delpath = argv[loop];
     if (!delpath) {
       PrintUsage();
       exit(1);
     }

     XrdOucString delsrc(delpath);
     gettimeofday(&abs_start_time,&tz);
     
     if (authz) {
       // append the authz information to all root: protocol names
       if (delsrc.beginswith("root://")) {
	 if (delsrc.find('?') != STR_NPOS) {
	   delsrc += "&authz=";
	 } else {
	   delsrc += "?&authz=";
	 }
	 delsrc += (const char *)authz;
       }
     }
     
     if (delsrc.beginswith("root://")) {
       long id, flags, modtime;
       long long size;
       
       XrdOucString url(delsrc);
       XrdClientUrlInfo u(url);
       
       XrdClientAdmin* client = new XrdClientAdmin(url.c_str());
       if (!client) {
	 CERR(("xrm: error creating admin client\n"));
	 exit(-3);
       }
       
       if (!client->Connect()) {
	 CERR(("xrm: error connecting to remote xrootd: %s\n",url.c_str()));
	 exit(-3);
       }
       
       
       if ( ( !client->Stat(u.File.c_str(), id, size, flags, modtime ))) {
	 CERR(("xrm: error remote source file does not exist: %s\n",delsrc.c_str()));
	 
	 delete client;
	 client =0;
	 continue;
       }
       
       if (client->GetCurrentUrl().IsValid()) {
	 u.Host = client->GetCurrentUrl().Host;
	 u.Port = client->GetCurrentUrl().Port;
	 url = u.GetUrl();
       }
       
       if ( !client->Rm(u.File.c_str())) {
	 CERR(("xrm: error removing remote file: %s\n",delsrc.c_str()));
	 delete client;
	 continue;
       }
       retval = 0;
       delete client;
     } else {
       struct stat stat_buf;
       int dostat = stat(delsrc.c_str(), &stat_buf);
       if (dostat) {
	 
	 CERR(("xrm: error: file %s does not exist!",delsrc.c_str()));
	 continue;
       }
       
       int dorm = unlink(delsrc.c_str());
       if (dorm) {
	 CERR(("xrm: error: no permissions to remove %s !",delsrc.c_str()));
	 continue;
       }
       retval = 0;
     }
     
     if (summary) {
       print_summary(delsrc.c_str());
     }
   }
   return retval;
}
