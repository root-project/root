//         $Id$

const char *XrdClientPrepCVSID = "$Id$";

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
   char c, Target[512], buff[16384], *bp, *sp, *theBuff = buff;
   char *inFile = 0;
   kXR_char Prty = 0, Opts = 0;
   long Debug = 0;
   int setDebug = 0, didPrep = 0, theBsz = sizeof(buff)-2, bsz, slen, rc;

// Process the options
//
   opterr = 0;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc,argv,"d:f:p:sStw")) && ((unsigned char)c != 0xff))
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
       case 'S': Opts |=(kXR_stage|kXR_coloc);
                 break;
       case 't': Opts |= kXR_fresh;
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

// If an infile was specified, make sure we can open it
//
   if (inFile && !(Stream = fopen(inFile, "r")))
      {cerr <<"xprep: " <<strerror(errno) <<" opening " <<inFile <<endl;
       exit(4);
      }

// If co0=location wanted, tyhen we must save the first file
//
   if (Opts & kXR_coloc && optind < argc)
      {sp = argv[optind]; slen = strlen(sp);
       strcpy(buff+1, sp); theBuff += slen+1; theBsz -= (slen+1); optind++;
      }

// Pre-process any command line paths at this point
//
   do {bp = theBuff; bsz = theBsz;
       while(optind < argc)
            {sp = argv[optind]; slen = strlen(sp);
             if (bsz <= slen) break;
             *bp++ = '\n';      strcpy(bp, sp); bp += slen; bsz -= (slen+1);
             optind++;
            }
       if (bp == theBuff) break;
       if (!Admin->Prepare(buff+1, Opts, Prty)) Fatal("Prepare", Admin);
       didPrep = 1;
      } while(optind < argc);

// If colocating, make sure we have the anchor file
//
   if (Opts & kXR_coloc && theBuff == buff)
      {if (!Stream || !(sp = fgets(buff+1, MaxPathLen, Stream))) inFile = 0;
          else {slen = strlen(sp); theBsz -= (slen+1); theBuff += slen+1;}
      } else theBuff++;

// Process the file
//
   if (inFile)
      {do {bp = theBuff; bsz = theBsz;
           while(bsz >= MaxPathLen)
                {if (!(sp = fgets(bp, MaxPathLen, Stream))) break;
                    {slen = strlen(sp); bsz -= slen; bp += slen;}
                }
           if (bp == theBuff) break;
           if (!Admin->Prepare(buff+1, Opts, Prty)) Fatal("Prepare", Admin);
           didPrep = 1;
          } while(!feof(Stream) && !ferror(Stream));
       if ((rc = ferror(Stream)))
          {cerr <<"xprep: Error " <<rc <<" reading " <<inFile <<endl;
           exit(4);
          }
       fclose(Stream);
      }

// If coloc is active, make sure we actually did a prepare
//
   if (!didPrep)
      {if (theBuff > buff+1)
          {*theBuff = '\0';
           if (!Admin->Prepare(buff+1, Opts, Prty)) Fatal("Prepare", Admin);
          } else {cerr <<"xprep: No files to prepare were specified" <<endl;
                  Usage(1);
                 }
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
cerr <<"\nOptions: [-d n] [-f fn] [-p prty] [-s] [-S] [-w]" <<endl;
exit(rc);
}
