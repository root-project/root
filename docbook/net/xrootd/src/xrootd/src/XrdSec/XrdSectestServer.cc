/******************************************************************************/
/*                                                                            */
/*                   X r d S e c t e s t S e r v e r . c c                    */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

const char *XrdSectestServerCVSID = "$Id$";

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/param.h>
#include <sys/socket.h>

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSec/XrdSecInterface.hh"
  
/******************************************************************************/
/*                    L O C A L   D E F I N I T I O N S                       */
/******************************************************************************/

#define H(x)         fprintf(stderr,x); fprintf(stderr, "\n");
#define I(x)         fprintf(stderr, "\n"); H(x)
#define insx(a,b)    sprintf(errbuff,a,b)
#define insy(a,b,c)  sprintf(errbuff,a,b,c)

typedef unsigned char uchar;

/******************************************************************************/
/*                      g l o b a l   v a r i a b l e s                       */
/******************************************************************************/

/* Define the execution control structure.
*/
struct myOpts {
  int  debug;            /* 1 -> Enable debugging.                   */
  int  bin;              /* 1 -> Input cred in binary format.        */
  int  xtra;             /* 1 -> Perform null cred test              */
  int  online;           /* 1 -> Filename is actual hex cred.        */
  char *cfn;             /* -> config file                           */
  char *host;            /* -> hostname                              */
  char *inpt;            /* -> Input stream name.                    */
  FILE *infid;           /* -> Input stream (normally stdin).        */
  } opts;

/* Define global variables.
*/
char errbuff[256];
#ifndef C_Block
char hexbuff[256];
#else
char hexbuff[sizeof(C_Block)+8];
#endif

/******************************************************************************/
/*                  f u n c t i o n   d e f i n i t i o n s                   */
/******************************************************************************/

int getbintix(uchar *buff, int blen);
void getargs(int argc, char **argv);
int  unhex(uchar *ibuff, uchar *obuff, int blen);
int  cvtx(uchar idig, uchar *odig);
void getline(uchar *buff, int blen);
char *Ereason( );
int emsg(int rc,char *msg);
void help(int rc);
void xerr(int x);

/******************************************************************************/
/*                          M A I N   P R O G R A M                           */
/******************************************************************************/

int main(int argc, char **argv)
{
  int i, rc;
  struct sockaddr    caddr;
  struct sockaddr_in *netaddr = (struct sockaddr_in *)&caddr;

  XrdOucErrInfo einfo;
  XrdSysLogger Logger;
  XrdSecService *ServerSecurity;
  XrdSecParameters *parmp;
  XrdSecCredentials cred((char *)malloc(8192), 8192);
  XrdSecProtocol *pp;
  unsigned char bbuff[4096];

// Parse the argument list.
//
   getargs(argc, argv);

// if hostname given, get the hostname address
//
   if (opts.host)
      {struct hostent *hp;
       if (!(hp = gethostbyname(opts.host)))
          {cerr <<"testServer: host '" <<opts.host <<"' not found." <<endl;
           exit(1);
          }
       memcpy((void *)&netaddr->sin_addr.s_addr, hp->h_addr_list[0],
              sizeof(netaddr->sin_addr.s_addr));
      }
      else {netaddr->sin_family = AF_INET;
            netaddr->sin_port   = 0;
            netaddr->sin_addr.s_addr = 0x80000001;
            opts.host = (char *)"localhost";
           }

// Create a new security server
//
   ServerSecurity = XrdSecgetService(&Logger, opts.cfn);
   if (!ServerSecurity) 
      {cerr <<"testServer: Unable to create server." <<endl; exit(1);}

// Get the security token and display it
//
   const char *sect = ServerSecurity->getParms(i, opts.host);
   if (!sect) cerr <<"testServer: No security token for " <<opts.host <<endl;
      else cerr <<"testServer: " <<i <<" bytes of SecToken='" <<sect <<"'" <<endl;

//Get the credentials from whatever source was specified
//
  if (opts.bin) cred.size = getbintix((uchar *)cred.buffer, cred.size);
     else {if (opts.online) strcpy((char *)bbuff, opts.inpt);
               else getline(bbuff, sizeof(bbuff));
           cred.size = unhex(bbuff, (uchar *)cred.buffer, cred.size);
          }

// Verify the length
//
   if (cred.size < 0) emsg(100,(char *)"Invalid credentials format.");

// Get the protocol
//
   if (!(pp = ServerSecurity->getProtocol(opts.host,
                                          (const sockaddr &)caddr,
                                          (const XrdSecCredentials *)&cred,
                                          &einfo)))
      {rc = einfo.getErrInfo();
       cerr << "testServer: getProtocol error " <<rc <<"; ";
       cerr  <<einfo.getErrText() <<endl;
       exit(1);
      }

// Now convert the credentials
//
   if (pp->Authenticate(&cred, &parmp, &einfo) < 0)
      {rc = einfo.getErrInfo();
       cerr << "testServer: Authenticate error " <<rc <<"; ";
       cerr  <<einfo.getErrText() <<endl;
       exit(1);
      }

// Tell everyone what the client identity is.
//
      cout <<(pp->Entity.name ? pp->Entity.name : "?")
           <<"@" <<(pp->Entity.host ? pp->Entity.host : "?")
           <<" prot=" <<pp->Entity.prot <<endl;

// All done
//
   exit(0);
}

/*getbintix: get binary credentials into an array.
*/
int getbintix(uchar *buff, int blen) {
int i, j;
    for (i = 0; i < blen; i++)
        if ((j = getc(opts.infid)) >= 0) buff[i] = (uchar)j;
           else if (j == EOF) return i;
                else xerr(insx("Error reading cred; %s.", Ereason()));
    xerr(insx("Cred longer than %d bytes.", blen));
    return -1;
}

/******************************************************************************/
/*                        Command Line Processing                             */
/******************************************************************************/

/* getargs: parse through argv obtaining options and parameters.
*/
void getargs(int argc, char **argv)
  {
  extern int optind; extern char *optarg; char c;

/* Establish defaults here.
*/
  opts.debug    = 0;
  opts.bin      = 0;
  opts.online   = 0;
  opts.cfn      = 0;
  opts.host     = 0;
  opts.xtra     = 0;
  opts.inpt     = (char *)"";
  opts.infid    = stdin;
  opts.cfn      = 0;

/* Process the options
*/
while ((c=getopt(argc,argv,"c:h:i:k:p:bdx")) != (char)EOF)
  { switch(c)
    {
    case 'b': opts.bin = 1;                            break;
    case 'c': opts.cfn  = optarg;                      break;
    case 'd': opts.debug = 1;                          break;
    case 'h': opts.host = optarg;                      break;
    case 'i': opts.inpt = optarg;                      break;
    case 'x': opts.xtra = 1;                           break;
    case '?': help(1);
    }
  }

/*Get the credentials, if specified on the command line.
*/
if (optind < argc) {opts.inpt = argv[optind++]; opts.online = 1;}

/*Make sure no more parameters exist.
*/
if (optind < argc) xerr(insx("Extraneous parameter, '%s'.", argv[optind]));

/*If the input stream is other than stdin, verify that it exists.
*/
if (opts.inpt[0] != '\000' && !opts.online
   && (!(opts.infid = fopen(opts.inpt, "r"))) )
   xerr(insy("Cannot open '%s'; %s.", opts.inpt, Ereason() ));

/* Make sure that -i * and -b are not specified together.
*/
if (opts.online && opts.bin) 
    emsg(8, (char *)"-b is incompatible with inline creds.");

/*All done
*/
  return;
    }

/******************************************************************************/
/*                          Utility  Function                                 */
/******************************************************************************/
  
/* unhex() converts a hex character string to its binary equivalent. The result
   is placed in the passed buffer. It returns the number of bytes extracted.
   An error results in a -1 response (including uneven hex digits). The
   input buffer must be terminated with a null.
*/
int  unhex(uchar *ibuff, uchar *obuff, int blen) {
int  i=0, j;
uchar dig1, dig2;

for (j = 0; j < blen; j++) {
  if (!ibuff[i]) return j;
  if (!cvtx(ibuff[i++], &dig1) || !cvtx(ibuff[i++], &dig2)) return -1;
  obuff[j] = (dig1 << 4) | dig2;
  }
return -1; /* Buffer overflow */
 }

int cvtx(uchar idig, uchar *odig) {
if (idig >= '0' && idig <= '9') {*odig = idig & (uchar)0x0f; return 1;}
idig = idig | (uchar)0x20; /* Change to lower case. */
if (idig < 'a' || idig > 'f') return 0;
*odig = (idig & (uchar)0x0f) + (uchar)0x09;
return 1;
}

/*getline() gets a newline terminated string from the expected input source.
*/
void getline(uchar *buff, int blen) {
  int i;
  if (!fgets((char *)buff, blen, opts.infid)) return;
  for (i = 0; i < blen; i++)
      if (buff[i] == '\n') {buff[i] = '\000'; break;}
  return;
  }

char *Ereason( ) {
  return strerror(errno);
  }

/*xerr: print message on standard error using the errbuff as source of message.
*/
void xerr(int x) { emsg(8, errbuff); }

/*emsg: print message on standard error.
*/
int emsg(int rc,char *msg) {
    cerr << "testServer: " <<msg <<endl;
    if (!rc) return 0;
    exit(rc);
    }

/*help prints hout the obvious.
*/
void help(int rc) {
/* Use H macro to avoid Sun string catenation bug. */
I("Syntax:   testServer [ options ] cred ]")
I("Options:  -b -c config -d  -h -i input -t")
H("          -p principal[.instance][@realm] -s sep")
I("Function: Display the credentials contents.")

if (rc > 1) exit(rc);
I("options:  (defaults: -k /etc/srvtab\\n")
I("-b        indicates the cred is in binary format (i.e., not hexchar).")
I("-c cfn    the config file.")
I("-d        turns on debugging.")
I("-h host   the incomming hostname.")
I("-i input  specifies the input stream (e.g., fname) if other than stdin.")
H("          This -i is ignored if cred is specified on the command line.")
exit(rc);
}
