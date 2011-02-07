/******************************************************************************/
/*                                                                            */
/*                     X r d A p p s C c o n f i g . c c                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucNList.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                                i n L i s t                                 */
/******************************************************************************/
  
int inList(const char *var, const char **Vec)
{
   int i = 0;
   while(Vec[i] && strcmp(Vec[i],var)) i++;
   return (Vec[i] != 0);
}

/******************************************************************************/
/*                                 U s a g e                                  */
/******************************************************************************/
  
void Usage(int rc)
{
   cerr <<"\n Usage: cconfig -c <cfn> [-h <host>] [-n <name>] [-x <prog>] [<args>]"
          "\n<args>: [[pfx]*]<directive> | <directive>[*[sfx]] [<args>]" <<endl;
   exit(rc);
}

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   static const char *Pgm = "cconfig: ";
   extern char *optarg;
   extern int opterr, optind, optopt;

   XrdSysLogger       Logger;
   XrdSysError        Say(&Logger, "cconfig");
   XrdOucNList_Anchor DirQ;
   XrdOucEnv          myEnv, *oldEnv = 0;
   XrdOucStream      *Config;

   const char *Cfn = 0, *Host = 0, *Name = 0, *Xeq = "xrootd";
   const char *noSub[] = {"cms.prepmsg", "ofs.notifymsg", "oss.stagemsg",
                          "frm.xfr.copycmd", 0};
   const char *ifChk[] = {"xrd.port", "all.role", "all.manager", 0};
   const char *slChk[] = {"frm.xfr.copycmd", 0};

   char buff[4096], *var, c;
   int i, retc = 0, cfgFD, chkQ = 0;

// Process the options
//
   opterr = 0;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc,argv,":c:h:n:x:")) && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'c': Cfn = optarg;
                 break;
       case 'h': Host= optarg;
                 break;
       case 'n': Name= optarg;
                 break;
       case 'x': Xeq = optarg;
                 break;
       default:  sprintf(buff,"'%c'", optopt);
                 if (c == ':') Say.Say(Pgm, buff, " value not specified.");
                    else Say.Say(Pgm, buff, " option is invalid.");
                 Usage(1);
       }
     }

// Make sure config file has been specified
//
   if (!Cfn) {Say.Say(Pgm, "Config file not specified."); Usage(1);}

// Get full host name
//
   if (!Host) Host = XrdNetDNS::getHostName();
      else {sockaddr IPAddr;
            Host = (XrdNetDNS::getHostAddr(Host, &IPAddr)
                 ?  XrdNetDNS::getHostName(IPAddr) : 0);
           }
   if (!Host) {Say.Say(Pgm, "Unable to determine host name."); exit(3);}

// Prepare all selector arguments
//
   for (i = optind; i < argc; i++) DirQ.Replace(argv[i],0);
   chkQ = (DirQ.First() != 0);

// Open the config file
//
   if ( (cfgFD = open(Cfn, O_RDONLY, 0)) < 0)
      {Say.Say(Pgm, strerror(errno), " opening config file ", Cfn);
       exit(4);
      }

// Construct instance name and stream
//
   Name = XrdOucUtils::InstName(Name);
   sprintf(buff,"%s %s@%s", Xeq, Name, Host);
   Config = new XrdOucStream(&Say, strdup(buff), &myEnv, "");
   Config->Attach(cfgFD);

// Now start reading records until eof.
//
   while((var = Config->GetMyFirstWord()))
        {if (chkQ && !DirQ.Find(var)) {Config->noEcho(); continue;}
              if (inList(var, noSub))
                 {if (inList(var, slChk))
                     while((var = Config->GetWord()) && *var != '/');
                  oldEnv = Config->SetEnv(0);
                  if (var) Config->GetRest(buff, sizeof(buff));
                  Config->SetEnv(oldEnv);
                 }
         else if (inList(var, ifChk))
                 {while((var = Config->GetWord()) && strcmp(var, "if"));
                  if (var && !XrdOucUtils::doIf(&Say, *Config, "directive",
                                                Host, Name, Xeq))
                     {Config->noEcho(); continue;}
                 }
         else Config->GetRest(buff, sizeof(buff));
         Config->Echo();
        }

// Now check if any errors occured during file i/o
//
   if ((retc = Config->LastError()))
      {Say.Say(Pgm, strerror(retc), " reading config file ", Cfn); retc = 8;}
   Config->Close();

// Should never get here
//
   exit(retc);
}
