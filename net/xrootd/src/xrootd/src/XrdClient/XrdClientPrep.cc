#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/types.h>

#include "XProtocol/XProtocol.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientEnv.hh"
  
/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char **argv)
{
   extern char *optarg;
   extern int  optind, opterr;
   extern void Fatal(const char *, XrdClientAdmin *);
   extern void Usage(int);
   static const int MaxPathLen = MAXPATHLEN+1;
   XrdClientAdmin *Admin;
   FILE *Stream = 0;
   char c, Target[512], buff[8192], *bp, *sp;
   char *inFile = 0;
   kXR_char Prty = 0, Opts = 0;
   long Debug = 0;
   int setDebug = 0, bsz, slen, rc;

// Process the options
//
   opterr = 0;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc,argv,"d:f:p:sw")) && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'd': Debug = atol(optarg); setDebug = 1;
                 break;
       case 'f': inFile = optarg;
                 break;
       case 'p': Prty = kXR_char(atoi(optarg));
                 break;
       case 's': Opts |= kXR_stage;
                 break;
       case 'w': Opts |= kXR_wmode;
                 break;
       default:  cerr <<"xprep: Invalid option '-" <<argv[optind-1] <<"'" <<endl;
                 Usage(1);
       }
     }

// Make sure a host has been specified
//
   if (optind >= argc || !isalnum(*argv[optind]))
      {cerr <<"xprep: target host name not specified" <<endl;
       Usage(1);
      }

// Get the target host
//
   strcpy(Target, "root://");
   strcat(Target, argv[optind]);
   strcat(Target, "//prep");
   optind++;

// Initialize the the admin (we need only one)
//
// Establish debugging level
//
   EnvPutInt(NAME_DEBUG, (setDebug ? Debug : -1));
   Admin = new XrdClientAdmin(Target);
   EnvPutInt(NAME_DEBUG, (setDebug ? Debug :  0));
   if (!Admin->Connect()) Fatal("Connect", Admin);

// Make sure we have something to prepare
//
   if (optind >= argc && !inFile)
      {cerr <<"xprep: No files to prepare were specified" <<endl;
       Usage(1);
      }

// If an infile was specified, make sure we can open it
//
   if (inFile && !(Stream = fopen(inFile, "r")))
      {cerr <<"xprep: " <<strerror(errno) <<" opening " <<inFile <<endl;
       exit(4);
      }

// Pre-process any command line paths at this point
//
   do {bp = buff; bsz = sizeof(buff)-1; *buff = 0;
       while(optind < argc)
            {sp = argv[optind]; slen = strlen(sp);
             if (bsz <= slen) break;
             *bp++ = '\n';      strcpy(bp, sp); bp += slen;
             optind++;
            }
       if (!(*buff)) break;
       if (!Admin->Prepare(buff+1, Opts, Prty)) Fatal("Prepare", Admin);
      } while(optind < argc);

// Process the file
//
   if (inFile)
      {do {bp = buff; bsz = sizeof(buff)-1; *buff = 0;
           while(bsz >= MaxPathLen)
                {if (!(sp = fgets(bp, MaxPathLen, Stream))) break;
                    {slen = strlen(sp); bsz -= slen; bp += slen;}
                }
           if (!(*buff)) break;
           if (!Admin->Prepare(buff, Opts, Prty)) Fatal("Prepare", Admin);
          } while(!feof(Stream) && !ferror(Stream));
       if ((rc = ferror(Stream)))
          {cerr <<"xprep: Error " <<rc <<" reading " <<inFile <<endl;
           exit(4);
          }
       fclose(Stream);
      }

// All done
//
   exit(0);
}

/******************************************************************************/
/*                                 F a t a l                                  */
/******************************************************************************/

void Fatal(const char *What, XrdClientAdmin *theAdmin)
{
   char *etext = theAdmin->LastServerError()->errmsg;

// Print a message and exit
//
   if (etext && *etext) cerr <<"xprep: " <<What <<" failed; " <<etext <<endl;
      else              cerr <<"xprep: " <<What <<" failed"           <<endl;
   exit(16);
}

/******************************************************************************/
/*                                 U s a g e                                  */
/******************************************************************************/
  
void Usage(int rc)
{
cerr <<"\nUsage:   xprep [options] host[:port][,...] [path [...]]\n";
cerr <<"\nOptions: [-d n] [-f fn] [-p prty] [-s] [-w]" <<endl;
exit(rc);
}
