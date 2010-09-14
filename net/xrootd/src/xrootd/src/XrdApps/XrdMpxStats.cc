/******************************************************************************/
/*                                                                            */
/*                        X r d M p x S t a t s . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/*   $Id$           */

const char *XrdMpxStatsCVSID = "$Id$";

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/uio.h>

#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"
  
/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

namespace XrdMpx
{
       XrdSysLogger       Logger;

       XrdSysError        Say(&Logger, "mpxstats");

static const int          addSender = 0x0001;

       int                Opts;

       int                Debug;
};

using namespace XrdMpx;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
/******************************************************************************/
/*                             X r d M p x V a r                              */
/******************************************************************************/
  
class  XrdMpxVar
{
public:

      int   Pop(const char *vName);

      int   Push(const char *vName);

      void  Reset() {vEnd = vBuff; vNum = -1; *vBuff = 0;}

const char *Var() {return vBuff;}

            XrdMpxVar() : vFence(vBuff + sizeof(vBuff) - 1) {Reset();}
           ~XrdMpxVar() {}

private:

static const int   vMax = 15;
             char *vEnd, *vFence, *vStack[vMax+1], vBuff[1024];
             int   vNum;
};

/******************************************************************************/
/*                        X r d M p x V a r : : P o p                         */
/******************************************************************************/
  
int XrdMpxVar::Pop(const char *vName)
{
    if (Debug) cerr <<"Pop:  " <<(vName ? vName : "") <<"; var=" <<vBuff <<endl;
    if (vNum < 0 || (vName && strcmp(vStack[vNum], vName))) return 0;
    vEnd = vStack[vNum]-1; *vEnd = '\0'; vNum--;
    return 1;
}

/******************************************************************************/
/*                       X r d M p x V a r : : P u s h                        */
/******************************************************************************/
  
int XrdMpxVar::Push(const char *vName)
{
   int n = strlen(vName);

   if (Debug) cerr <<"Push: " <<vName <<"; var=" <<vBuff <<endl;
   if (vNum >= vMax) return 0;
   if (vNum >= 0) *vEnd++ = '.';
      else         vEnd = vBuff;
   if (vEnd+n+1 >= vFence) return 0;
   strcpy(vEnd, vName);
   vStack[++vNum] = vEnd;
   vEnd += n;
   return 1;
}

/******************************************************************************/
/*                             X r d M p x X m l                              */
/******************************************************************************/
  
class XrdMpxXml
{
public:

enum fmtType {fmtCGI, fmtFlat, fmtXML};

int Format(const char *Host, char *ibuff, char *obuff);

    XrdMpxXml(fmtType ft) : fType(ft)
                          {if (ft == fmtCGI) {vSep = '='; vSfx = '&';}
                              else           {vSep = ' '; vSfx = '\n';}
                          }
   ~XrdMpxXml() {}

private:

struct VarInfo
      {const char *Name;
             char *Data;
      };

char *Add(char *Buff, const char *Var, const char *Val);
void  getVars(XrdOucTokenizer &Data, VarInfo Var[]);
int   xmlErr(const char *t1, const char *t2=0, const char *t3=0);

fmtType fType;
char    vSep;
char    vSfx;
};

/******************************************************************************/
/*                     X r d M p x X m l : : F o r m a t                      */
/******************************************************************************/
  
int XrdMpxXml::Format(const char *Host, char *ibuff, char *obuff)
{
   static const char *Hdr0 = "<statistics ";
   static const int   H0Len = strlen(Hdr0);

   XrdMpxVar       xVar;
   XrdOucTokenizer Data(ibuff);
   VarInfo vHead[] = {{"tod", 0}, {"ver", 0}, {"src", 0}, {"tos", 0},
                      {"pgm", 0}, {"ins", 0}, {"pid", 0}, {0, 0}};
   VarInfo vStat[] = {{"id",  0}, {0, 0}};
   VarInfo vTail[] = {{"toe", 0}, {0, 0}};
   char *lP = ibuff, *oP = obuff, *tP, *vP;
   int i, rc;

// Insert a newline for the first '>'
//
   if (!(lP = (char *)index(lP, '>')))
      return xmlErr("Invalid xml stream: ", ibuff);
   *lP++ = '\n';

// Now make the input tokenizable
//
   while(*lP)
        {if (*lP == '>' || (*lP == '<' && *(lP+1) == '/')) *lP = ' ';
         lP++;
        }

// The first token better be '<statistics'
//
   if (!(lP = Data.GetLine()) || strncmp(Hdr0, lP, H0Len))
      return xmlErr("Stream does not start with '<statistics'.");
   Data.GetToken(); getVars(Data, vHead);

// Output the vars in the headers as 'stats..var'
//
   for (i = 0; vHead[i].Name; i++)
       {if (vHead[i].Data) oP = Add(oP, vHead[i].Name, vHead[i].Data);}

// Add in the host name, if supplied
//
   if (Host) oP = Add(oP, "host", Host);

// Get the remainder
//
   if (!Data.GetLine()) return xmlErr("Null xml stream after header.");

// The following segment reads all of the "stats" entries
//
   while((tP = Data.GetToken()) && strcmp(tP, "/statistics"))
        {     if (*tP == '/')
                 {if (!xVar.Pop(strcmp("/stats", tP) ? tP+1 : 0))
                     return xmlErr(tP, "invalid end for ", xVar.Var());
                 }
         else if (*tP == '<')
                 {if (strcmp("<stats", tP)) rc = xVar.Push(tP+1);
                     else {getVars(Data, vStat);
                           rc = (vStat[0].Data ? xVar.Push(vStat[0].Data)
                                               : xVar.Push(tP+1));
                          }
                  if (!rc) return xmlErr("Nesting too deep for ", xVar.Var());
                 }
         else    {if ((vP = index(tP, '<'))) *vP = '\0';
                  if (*tP == '"')
                     {i = strlen(tP)-1;
                      if (*(tP+i) == '"') {*(tP+i) = '\0'; i = 1;}
                     } else i = 0;
                  oP = Add(oP, xVar.Var(), tP+i);
                  if (vP) {*vP = '<';
                           if (vP != tP) memset(tP, ' ', vP - tP);
                           Data.RetToken();
                          }
                 }
        }
   if (!tP) return xmlErr("Missing '</statistics>' in xml stream.");
   getVars(Data, vTail);
   if (vTail[0].Data) oP = Add(oP, vTail[0].Name, vTail[0].Data);
   if (*(oP-1) == '&') oP--;
   *oP++ = '\n';
   return oP - obuff;
}

/******************************************************************************/
/*                        X r d M p x X m l : : A d d                         */
/******************************************************************************/
  
char *XrdMpxXml::Add(char *Buff, const char *Var, const char *Val)
{
   strcpy(Buff, Var); Buff += strlen(Var);
   *Buff++ = vSep;
   strcpy(Buff, Val); Buff += strlen(Val);
   *Buff++ = vSfx;
   return Buff;
}

/******************************************************************************/
/*                                                                            */
/*                    X r d M p x X m l : : g e t V a r s                     */
/*                                                                            */
/******************************************************************************/
  
void XrdMpxXml::getVars(XrdOucTokenizer &Data, VarInfo Var[])
{
   char *tVar, *tVal;
   int i;

// Initialize the data pointers to null
//
   i = 0;
   while(Var[i].Name) Var[i++].Data = 0;

// Get all of the variables/values and return where possible
//
   while((tVar = Data.GetToken()) && *tVar != '<' && *tVar != '/')
        {if (!(tVal = (char *)index(tVar, '='))) continue;
         *tVal++ = '\0';
         if (*tVal == '"')
            {tVal++, i = strlen(tVal);
             if (*(tVal+i-1) == '"') *(tVal+i-1) = '\0';
            }
         i = 0;
         while(Var[i].Name)
              {if (!strcmp(Var[i].Name, tVar)) {Var[i].Data = tVal; break;}
                  else i++;
              }
        }
   if (tVar && (*tVar == '<' || *tVar == '/')) Data.RetToken();
}

/******************************************************************************/
/*                     X r d M p x X m l : : x m l E r r                      */
/******************************************************************************/
  
int XrdMpxXml::xmlErr(const char *t1, const char *t2, const char *t3)
{
   Say.Emsg(":", t1, t2, t3);
   return 0;
}
  
/******************************************************************************/
/*                             X r d M p x O u t                              */
/******************************************************************************/
  
class XrdMpxOut
{
public:

struct statsBuff
      {statsBuff      *Next;
       struct sockaddr From;
       int             Dlen;
       char            Data[8190];
       char            Pad[2];
     };

void       Add(statsBuff *sbP);

statsBuff *getBuff();

void      *Run(XrdMpxXml *xP);

           XrdMpxOut() : Ready(0), inQ(0), Free(0) {}
          ~XrdMpxOut() {}

private:

XrdSysMutex     myMutex;
XrdSysSemaphore Ready;

statsBuff      *inQ;
statsBuff      *Free;
};

/******************************************************************************/
/*                        X r d M p x O u t : : A d d                         */
/******************************************************************************/
  
void XrdMpxOut::Add(statsBuff *sbP)
{

// Add this to the queue and signal the processing thread
//
   myMutex.Lock();
   sbP->Next = inQ;
   inQ = sbP;
   Ready.Post();
   myMutex.UnLock();
}

/******************************************************************************/
/*                    X r d M p x O u t : : g e t B u f f                     */
/******************************************************************************/
  
XrdMpxOut::statsBuff *XrdMpxOut::getBuff()
{
   statsBuff *sbP;

// Use an available buffer or allocate one
//
   myMutex.Lock();
   if ((sbP = Free)) Free = sbP->Next;
      else sbP = new statsBuff;
   myMutex.UnLock();
   return sbP;
}

/******************************************************************************/
/*                        X r d M p x O u t : : R u n                         */
/******************************************************************************/
  
void *XrdMpxOut::Run(XrdMpxXml *xP)
{
   char *bP, *Host=0, obuff[sizeof(statsBuff)*2];
   statsBuff *sbP;
   int wLen, rc;

// Simply loop formating and outputing the buffers
//
   while(1)
        {Ready.Wait();
         myMutex.Lock();
         if ((sbP = inQ)) inQ = sbP->Next;
         myMutex.UnLock();
         if (!sbP) continue;
         if (xP)
            {Host = (Opts & addSender ? XrdNetDNS::getHostName(sbP->From) : 0);
             wLen = xP->Format(Host, sbP->Data, obuff);
             bP = obuff;
             if (Host) free(Host);
            } else {
             bP = sbP->Data;
             *(bP + sbP->Dlen) = '\n';
             wLen = sbP->Dlen+1;
            }

         while(wLen > 0)
              {do {rc = write(STDOUT_FILENO, bP, wLen);}
                  while(rc < 0 && errno == EINTR);
               wLen -= rc; bP += rc;
              }

         myMutex.Lock(); sbP->Next = Free; Free = sbP; myMutex.UnLock();
        }

// Should never get here
//
   return (void *)0;
}

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/
  
namespace XrdMpx
{
XrdMpxOut statsQ;
};

/******************************************************************************/
/*                     T h r e a d   I n t e r f a c e s                      */
/******************************************************************************/
  
void *mainOutput(void *parg)
{
    XrdMpxXml *xP = static_cast<XrdMpxXml *>(parg);
    return statsQ.Run(xP);
}

/******************************************************************************/
/*                                 U s a g e                                  */
/******************************************************************************/
  
void Usage(int rc)
{
   cerr <<"\nUsage: mpxstats [-f {cgi|flat|xml}] -p <port> [-s]" <<endl;
   exit(rc);
}

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   extern char *optarg;
   extern int opterr, optopt;
   sigset_t myset;
   pthread_t tid;
   XrdMpxXml::fmtType fType = XrdMpxXml::fmtXML;
   XrdMpxOut::statsBuff *sbP = 0;
   XrdNetSocket mySocket(&Say);
   XrdMpxXml *xP = 0;
   SOCKLEN_t fromLen;
   int Port = 0, retc, udpFD;
   char buff[64], c;

// Process the options
//
   opterr = 0; Debug = 0; Opts = 0;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc,argv,"df:p:s")) && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'd': Debug = 1;
                 break;
       case 'f':      if (!strcmp(optarg, "cgi" )) fType = XrdMpxXml::fmtCGI;
                 else if (!strcmp(optarg, "flat")) fType = XrdMpxXml::fmtFlat;
                 else if (!strcmp(optarg, "xml" )) fType = XrdMpxXml::fmtXML;
                 else {Say.Emsg(":", "Invalid format - ", optarg); Usage(1);}
                 break;
       case 'h': Usage(0);
       case 'p': if (!(Port = atoi(optarg)))
                    {Say.Emsg(":", "Invalid port number - ", optarg); Usage(1);}
                 break;
       case 's': Opts |= addSender;
                 break;
       default:  sprintf(buff,"'%c'", optopt);
                 if (c == ':') Say.Emsg(":", buff, "value not specified.");
                    else Say.Emsg(0, buff, "option is invalid");
                 Usage(1);
       }
     }

// Make sure port has been specified
//
   if (!Port) {Say.Emsg(":", "Port has not been specified."); Usage(1);}

// Turn off sigpipe and host a variety of others before we start any threads
//
   signal(SIGPIPE, SIG_IGN);  // Solaris optimization
   sigemptyset(&myset);
   sigaddset(&myset, SIGPIPE);
   sigaddset(&myset, SIGCHLD);
   pthread_sigmask(SIG_BLOCK, &myset, NULL);

// Set the default stack size here
//
   if (sizeof(long) > 4) XrdSysThread::setStackSize((size_t)1048576);
      else               XrdSysThread::setStackSize((size_t)786432);

// Create a UDP socket and bind it to a port
//
   if (mySocket.Open(0, Port, XRDNET_SERVER|XRDNET_UDPSOCKET, 0) < 0)
      {Say.Emsg(":", -mySocket.LastError(), "create udp socket"); exit(4);}
   udpFD = mySocket.Detach();

// Establish format
//
   if (fType != XrdMpxXml::fmtXML) xP = new XrdMpxXml(fType);

// Now run a thread to output whatever we get
//
   if ((retc = XrdSysThread::Run(&tid, mainOutput, (void *)xP,
                                 XRDSYSTHREAD_BIND, "Output")))
      {Say.Emsg(":", retc, "create output thread"); exit(4);}

// Now simply wait for the messages
//
   while(1)
        {sbP = statsQ.getBuff();
         fromLen = sizeof(sbP->From);
         retc = recvfrom(udpFD, sbP->Data, sizeof(sbP->Data), 0,
                               &sbP->From, &fromLen);
         if (retc < 0) {Say.Emsg(":", retc, "recv udp message"); exit(8);}
         sbP->Dlen = retc;
         statsQ.Add(sbP);
        }

// Should never get here
//
   return 0;
}
