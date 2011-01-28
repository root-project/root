//////////////////////////////////////////////////////////////////////////
//                                                                      //
// xrdcp                                                                //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
//                                                                      //
// A cp-like command line tool for xrootd environments                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

const char *XrdcpCVSID = "$Id$";

#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdCpMthrQueue.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdCpWorkLst.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdSys/XrdSysPlatform.hh"

#include <XrdCrypto/XrdCryptoFactory.hh>
#include <XrdCrypto/XrdCryptoMsgDigest.hh>

#include "XrdClient/XrdClientAbsMonIntf.hh"
#include "XrdClient/XrdcpXtremeRead.hh"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#include <dlfcn.h>
#endif
#include <stdarg.h>
#include <stdio.h>

#ifdef HAVE_LIBZ
#include <zlib.h>
#endif

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

#define COUT(s) do {				\
      cout_print s;				\
   } while (0)

#define CERR(s) do {				\
      cerr_print s;				\
   } while (0)

}
//////////////////////////////////////////////////////////////////////


struct XrdCpInfo {
   XrdClient                    *XrdCli;
   int                          localfile;
   long long                    len, bread, bwritten;
   XrdCpMthrQueue               queue;
   XrdClientAbsMonIntf          *mon;
} cpnfo;

#define XRDCP_BLOCKSIZE          (8*1024*1024)
#define XRDCP_XRDRASIZE          (30*XRDCP_BLOCKSIZE)
#define XRDCP_VERSION            "(C) 2004-2010 by the Xrootd group. $Revision: 1.103 $ - Xrootd version: "XrdVSTRING

///////////////////////////////////////////////////////////////////////
// Coming from parameters on the cmd line

bool summary=false;            // print summary
bool progbar=true;             // print progbar
bool md5=false;                // print md5
bool adlerchk=false;           // print adler32 chksum

XrdOucString monlibname = "libXrdCpMonitorClient.so"; // Default name for the ext monitoring lib

char *srcopaque=0,
   *dstopaque=0;   // opaque info to be added to urls
// Default open flags for opening a file (xrd)
kXR_unt16 xrd_wr_flags=kXR_async | kXR_mkpath | kXR_open_updt | kXR_new;

// Flags for open() to force overwriting or not. Default is not.
#define LOC_WR_FLAGS_FORCE ( O_CREAT | O_WRONLY | O_TRUNC | O_BINARY );
#define LOC_WR_FLAGS       ( O_CREAT | O_WRONLY | O_EXCL | O_BINARY );
int loc_wr_flags = LOC_WR_FLAGS;

bool recurse = false;

char BWMHost[1024]; // The given bandwidth limiter on the local site. If not empty then a bwm has to be used

bool doXtremeCp = false;
XrdOucString XtremeCpRdr;

///////////////////////

// To compute throughput etc
struct timeval abs_start_time;
struct timeval abs_stop_time;
struct timezone tz;

#ifdef HAVE_XRDCRYPTO
// To calculate md5 sums during transfers
XrdCryptoMsgDigest *MD_5=0;    // md5 computation
XrdCryptoFactory *gCryptoFactory = 0;
#endif

// To calculate the adler32 cksum
unsigned int adler = 0;

#ifdef HAVE_XRDCRYPTO
void print_summary(const char* src, const char* dst, unsigned long long bytesread, XrdCryptoMsgDigest* _MD_5, unsigned int adler ) {
#else
void print_summary(const char* src, const char* dst, unsigned long long bytesread, unsigned int adler ) {
#endif
   gettimeofday (&abs_stop_time, &tz);
   float abs_time=((float)((abs_stop_time.tv_sec - abs_start_time.tv_sec) *1000 +
			   (abs_stop_time.tv_usec - abs_start_time.tv_usec) / 1000));


   XrdOucString xsrc(src);
   XrdOucString xdst(dst);
   xsrc.erase(xsrc.rfind('?'));
   xdst.erase(xdst.rfind('?'));

   COUT(("[xrdcp] #################################################################\n"));
   COUT(("[xrdcp] # Source Name              : %s\n",xsrc.c_str()));
   COUT(("[xrdcp] # Destination Name         : %s\n",xdst.c_str()));
   COUT(("[xrdcp] # Data Copied [bytes]      : %lld\n",bytesread));
   COUT(("[xrdcp] # Realtime [s]             : %f\n",abs_time/1000.0));
   if (abs_time > 0) {
      COUT(("[xrdcp] # Eff.Copy. Rate[MB/s]     : %f\n",bytesread/abs_time/1000.0));
   }
#ifdef HAVE_XRDCRYPTO
#ifndef WIN32
   if (md5) {
     COUT(("[xrdcp] # md5                      : %s\n",_MD_5->AsHexString()));
   }
#endif
#endif
   if (adlerchk) {
      COUT(("[xrdcp] # adler32                  : %x\n", adler));
   }
   COUT(("[xrdcp] #################################################################\n"));
}

void print_progbar(unsigned long long bytesread, unsigned long long size) {
   CERR(("[xrootd] Total %.02f MB\t|",(float)size/1024/1024));
   for (int l=0; l< 20;l++) {
      if (l< ( (int)(20.0*bytesread/size)))
	 CERR(("="));
      if (l==( (int)(20.0*bytesread/size)))
	 CERR((">"));
      if (l> ( (int)(20.0*bytesread/size)))
	 CERR(("."));
   }
  
   float abs_time=((float)((abs_stop_time.tv_sec - abs_start_time.tv_sec) *1000 +
			   (abs_stop_time.tv_usec - abs_start_time.tv_usec) / 1000));
   CERR(("| %.02f %% [%.01f MB/s]\r",100.0*bytesread/size,bytesread/abs_time/1000.0));
}

#ifdef HAVE_XRDCRYPTO
void print_chksum(const char* src, unsigned long long bytesread, XrdCryptoMsgDigest* _MD_5, unsigned adler) {
  if (_MD_5 || adlerchk) {
#else
void print_chksum(const char* src, unsigned long long bytesread, unsigned adler) {
  if (adlerchk) {
#endif
    XrdOucString xsrc(src);
    xsrc.erase(xsrc.rfind('?'));
    //    printf("md5: %s\n",_MD_5->AsHexString());
#ifdef HAVE_XRDCRYPTO
#ifndef WIN32
    if (_MD_5)
       cout << "md5: " << _MD_5->AsHexString() << " " << xsrc << " " << bytesread << endl;
#endif
#endif
    if (adlerchk)
       cout << "adler32: " << hex << adler << " " << xsrc << bytesread << endl;

  }
}




// The body of a thread which reads from the global
//  XrdClient and keeps the queue filled
//____________________________________________________________________________
void *ReaderThread_xrd(void *)
{

   Info(XrdClientDebug::kHIDEBUG,
	"ReaderThread_xrd",
	"Reader Thread starting.");
   
   pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, 0);
   pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);


   void *buf;
   long long offs = 0;
   int nr = 1;
   long long bread = 0, len = 0;
   long blksize;

   len = cpnfo.len;

   while ((nr > 0) && (offs < len)) {
      buf = malloc(XRDCP_BLOCKSIZE);
      if (!buf) {
	 cerr << "Out of memory." << endl;
	 abort();
      }

      
      blksize = xrdmin(XRDCP_BLOCKSIZE, len-offs);

      if ( (nr = cpnfo.XrdCli->Read(buf, offs, blksize)) ) {
         cpnfo.queue.PutBuffer(buf, offs, nr);
         cpnfo.XrdCli->RemoveDataFromCache(offs, offs+nr-1, false);
	 bread += nr;
	 offs += nr;
      }

      pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
      pthread_testcancel();
      pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
   }

   cpnfo.bread = bread;

   // This ends the transmission... bye bye
   cpnfo.queue.PutBuffer(0, 0, 0);

   return 0;
}




// The body of a thread which reads from the global
//  XrdClient and keeps the queue filled
// This is the thread for extreme reads, in this case we may have multiple of these
// threads, reading the same file from different server endpoints
//____________________________________________________________________________
struct xtreme_threadnfo {
   XrdXtRdFile *xtrdhandler;

   // The client used by this thread
   XrdClient *cli;

   // A unique integer identifying the client instance
   int clientidx;

   // The block from which to start prefetching/reading
   int startfromblk;

   // Max convenient number of outstanding blks
   int maxoutstanding;
}; 
void *ReaderThread_xrd_xtreme(void *parm)
{

   Info(XrdClientDebug::kHIDEBUG,
	"ReaderThread_xrd_xtreme",
	"Reader Thread starting.");
   
   pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, 0);
   pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);

   void *buf;

   int nr = 1;
   int noutstanding = 0;


   // Which block to read
   XrdXtRdBlkInfo *blknfo = 0;
   xtreme_threadnfo *thrnfo = (xtreme_threadnfo *)parm;

   // Block to prefetch
   int lastprefetched = thrnfo->startfromblk;
   int lastread = lastprefetched;

   thrnfo->cli->SetCacheParameters(XRDCP_BLOCKSIZE*4*thrnfo->maxoutstanding*2, 0, XrdClientReadCache::kRmBlk_FIFO);
   if (thrnfo->cli->IsOpen_wait())
   while (nr > 0) {

      // Keep always some blocks outstanding from the point of view of this reader
      while (noutstanding < thrnfo->maxoutstanding) {
         int lp;
         lp = thrnfo->xtrdhandler->GetBlkToPrefetch(lastprefetched, thrnfo->clientidx, blknfo);
         if (lp >= 0) {
            //cout << "cli: " << thrnfo->clientidx << " prefetch: " << lp << " offs: " << blknfo->offs << " len: " << blknfo->len << endl;
            if ( thrnfo->cli->Read_Async(blknfo->offs, blknfo->len) == kOK ) {  
               lastprefetched = lp;
               noutstanding++;
            }
            else break;
         }
         else break;
      }

      int lr = thrnfo->xtrdhandler->GetBlkToRead(lastread, thrnfo->clientidx, blknfo);
      if (lr >= 0) {

         buf = malloc(blknfo->len);
         if (!buf) {
            cerr << "Out of memory." << endl;
            abort();
         }

         //cout << "cli: " << thrnfo->clientidx << "     read: " << lr << " offs: " << blknfo->offs << " len: " << blknfo->len << endl;

         // It is very important that the search for a blk to read starts from the first block upwards
         nr = thrnfo->cli->Read(buf, blknfo->offs, blknfo->len);
         if ( nr >= 0 ) {
            lastread = lr;
            noutstanding--;

            // If this block was stolen by somebody else then this client has to be penalized
            // If this client stole the blk to some other client, then this client has to be rewarded
            int reward = thrnfo->xtrdhandler->MarkBlkAsRead(lr);
            if (reward >= 0) 
               // Enqueue the block only if it was not already read
               cpnfo.queue.PutBuffer(buf, blknfo->offs, nr);

            if (reward > 0) {
               thrnfo->maxoutstanding++;
               thrnfo->maxoutstanding = xrdmin(20, thrnfo->maxoutstanding);
               thrnfo->cli->SetCacheParameters(XRDCP_BLOCKSIZE*4*thrnfo->maxoutstanding*2, 0, XrdClientReadCache::kRmBlk_FIFO);
            }
            if (reward < 0) {
               thrnfo->maxoutstanding--;
               free(buf);
            }

            if (thrnfo->maxoutstanding <= 0) {
               sleep(1);
               thrnfo->maxoutstanding = 1;
            }

         }

         // It is very important that the search for a blk to read starts from the first block upwards
         thrnfo->cli->RemoveDataFromCache(blknfo->offs, blknfo->offs+blknfo->len-1, false);
      }
      else {

         if (thrnfo->xtrdhandler->AllDone()) break;
         pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
         sleep(1);
      }


      pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
      pthread_testcancel();
      pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
   }

   // We get here if there are no more blocks to read or to steal from other readers
   // This ends the transmission... bye bye
   cpnfo.queue.PutBuffer(0, 0, 0);

   return 0;
}


// The body of a thread which reads from the global filehandle
//  and keeps the queue filled
//____________________________________________________________________________
void *ReaderThread_loc(void *) {

   Info(XrdClientDebug::kHIDEBUG,
	"ReaderThread_loc",
	"Reader Thread starting.");

   pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, 0);
   pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);

   void *buf;
   long long offs = 0;
   int nr = 1;
   long long bread = 0;

   while (nr > 0) {
      buf = malloc(XRDCP_BLOCKSIZE);
      if (!buf) {
	 cerr << "Out of memory." << endl;
	 abort();
      }

      if ( (nr = read(cpnfo.localfile, buf, XRDCP_BLOCKSIZE)) ) {
         cpnfo.queue.PutBuffer(buf, offs, nr);
	 bread += nr;
	 offs += nr;
      }
   }

   cpnfo.bread = bread;

   // This ends the transmission... bye bye
   cpnfo.queue.PutBuffer(0, 0, 0);

   return 0;
}


int CreateDestPath_loc(XrdOucString path, bool isdir) {
   // We need the path name without the file
   if (!isdir) {
      int pos = path.rfind('/');

      if (pos != STR_NPOS)
	 path.erase(pos);
      else path = "";


   }

   if (path != "")
      return ( MAKEDIR(
		     path.c_str(),
		     S_IRUSR | S_IWUSR | S_IXUSR |
		     S_IRGRP | S_IWGRP | S_IXGRP |
		     S_IROTH | S_IXOTH)
	       );
   else
      return 0;

}
   
void BuildFullDestFilename(XrdOucString &src, XrdOucString &dest, bool destisdir) {
   if (destisdir) {
      // We need the filename from the source
      XrdOucString fn(src);
      fn.erase(fn.find('?'));
      int lsl = fn.rfind('/');
      if (lsl != STR_NPOS)
         fn.erase(0, lsl+1);
      dest += fn;
   }
}

int CreateDestPath_xrd(XrdOucString url, bool isdir) {
   // We need the path name without the file
   bool statok = FALSE, done = FALSE, direxists = TRUE;
   long id, flags, modtime;
   long long size;
   char *path, *slash;

   if (url == "-") return 0;

   //   if (!isdir)
   url.erase(url.rfind('/') + 1);

   XrdClientAdmin *adm = new XrdClientAdmin(url.c_str());
   if (adm->Connect()) {
     XrdClientUrlInfo u(url);

     statok = adm->Stat((char *)u.File.c_str(), id, size, flags, modtime);

     // We might have been redirected to a destination server. Better to remember it and use
     //  only this one as output.
     if (adm->GetCurrentUrl().IsValid()) {
	u.Host = adm->GetCurrentUrl().Host;
	u.Port = adm->GetCurrentUrl().Port;
	url = u.GetUrl();
     }

     path = (char *)u.File.c_str();
     slash = path;

     // FIXME: drop the top level directory as it cannot be stat by the xrootd server
     slash += strspn(slash, "/");
     slash += strcspn(slash, "/");
     
     // If the path already exists, it's good
     done = (statok && (flags & kXR_isDir));

     // The idea of slash pointer is taken from the BSD mkdir implementation
     while (!done) {
       slash += strspn(slash, "/");
       slash += strcspn(slash, "/");
       
       char nextChar = *(slash+1);
       done = (*slash == '\0' || nextChar == '\0');
       *(slash+1) = '\0';

       if (direxists) {
	 statok = adm->Stat(path, id, size, flags, modtime);
	 if (!statok || (!(flags & kXR_xset) && !(flags & kXR_other))) {
	   direxists = FALSE;
	 }
       }
	 
       if (!direxists) {
	 Info(XrdClientDebug::kHIDEBUG,
	      "CreateDestPath__xrd",
	      "Creating directory " << path);
	 
	 adm->Mkdir(path, 7, 5, 5);
	 
       }
       *(slash+1) = nextChar;
     }
   }

   delete adm;
   return 0;
}

int doCp_xrd2xrd(XrdClient **xrddest, const char *src, const char *dst) {
   // ----------- xrd to xrd affair
   pthread_t myTID;
   XrdClientVector<pthread_t> myTIDVec;

   void *thret;
   XrdClientStatInfo stat;
   int retvalue = 0;

   gettimeofday(&abs_start_time,&tz);

   // Open the input file (xrdc)
   // If Xrdcli is non-null, the correct src file has already been opened
   if (!cpnfo.XrdCli) {
      cpnfo.XrdCli = new XrdClient(src);
      if ( ( !cpnfo.XrdCli->Open(0, kXR_async) ||
	     (cpnfo.XrdCli->LastServerResp()->status != kXR_ok) ) ) {
	 cerr << "Error opening remote source file " << src << endl;
	 PrintLastServerError(cpnfo.XrdCli);

	 delete cpnfo.XrdCli;
	 cpnfo.XrdCli = 0;
	 return 1;
      }
   }

   
   cpnfo.XrdCli->Stat(&stat);
   cpnfo.len = stat.size;
   
   // if xrddest if nonzero, then the file is already opened for writing
   if (!*xrddest) {
      *xrddest = new XrdClient(dst);
      
      if (!PedanticOpen4Write(*xrddest, kXR_ur | kXR_uw | kXR_gw | kXR_gr | kXR_or,
                              xrd_wr_flags)) {
         cerr << "Error opening remote destination file " << dst << endl;
         PrintLastServerError(*xrddest);
         
         delete cpnfo.XrdCli;
         delete *xrddest;
         *xrddest = 0;
         cpnfo.XrdCli = 0;
         return -1;
      }
      
   }
   
   // If the Extreme Copy flag is set, we try to find more sources for this file
   // Each source gets assigned to a different reader thread
   XrdClientVector<XrdClient *> xtremeclients;
   XrdXtRdFile *xrdxtrdfile = 0;
   
   if (doXtremeCp) 
      XrdXtRdFile::GetListOfSources(cpnfo.XrdCli, XtremeCpRdr, xtremeclients);
   
   // Start reader on xrdc
   if (doXtremeCp && (xtremeclients.GetSize() > 1)) {
      
      // Beware... with the extreme copy the normal read ahead mechanism
      // makes no sense at all.
      //EnvPutInt(NAME_REMUSEDCACHEBLKS, 1);
      xrdxtrdfile = new XrdXtRdFile(XRDCP_BLOCKSIZE*4, cpnfo.len);
      
      for (int iii = 0; iii < xtremeclients.GetSize(); iii++) {
         xtreme_threadnfo *nfo = new(xtreme_threadnfo);
         nfo->xtrdhandler = xrdxtrdfile;
         nfo->cli = xtremeclients[iii];
         nfo->clientidx = xrdxtrdfile->GimmeANewClientIdx();
         nfo->startfromblk = iii*xrdxtrdfile->GetNBlks() / xtremeclients.GetSize();
         nfo->maxoutstanding = xrdmin( 5, xrdxtrdfile->GetNBlks() / xtremeclients.GetSize() );

         XrdSysThread::Run(&myTID, ReaderThread_xrd_xtreme, (void *)nfo);
         myTIDVec.Push_back(myTID);
      }
      
   }
   else {
      XrdSysThread::Run(&myTID, ReaderThread_xrd, (void *)&cpnfo);
      myTIDVec.Push_back(myTID);
   }
   
   
   
   
   
   int len = 1;
   void *buf;
   long long offs = 0;
   long long bytesread=0;
   long long size = cpnfo.len;
   bool draining = false;
   
   // Loop to write until ended or timeout err
   while (1) {
      
      if (xrdxtrdfile && xrdxtrdfile->AllDone()) draining = true;
      if (draining && !cpnfo.queue.GetLength()) break;

      if ( cpnfo.queue.GetBuffer(&buf, offs, len) ) {

         if (len && buf) {

            bytesread+=len;
            if (progbar) {
               gettimeofday(&abs_stop_time,&tz);
               print_progbar(bytesread,size);
            }

#ifdef HAVE_XRDCRYPTO
            if (md5) {
               MD_5->Update((const char*)buf,len);
            }
#endif

#ifdef HAVE_LIBZ
            if (adlerchk) {
               adler = adler32(adler, (const Bytef*)buf, len);
            }
#endif

            if (!(*xrddest)->Write(buf, offs, len)) {
               cerr << "Error writing to output server." << endl;
               PrintLastServerError(*xrddest);
               retvalue = 11;
               break;
            }

            if (cpnfo.mon)
               cpnfo.mon->PutProgressInfo(bytesread, cpnfo.len, (float)bytesread / cpnfo.len * 100.0);

            free(buf);

         }
         else
            if (!xrdxtrdfile && ( ((buf == 0) && (len == 0)) || (bytesread >= size))) {
               if (buf) free(buf);
               break;
            }

      }
      else {
         cerr << endl << endl << 
            "Critical read timeout. Unable to read data from the source." << endl;
         retvalue = -1;
         break;
      }

      buf = 0;
   }

   if (cpnfo.mon)
      cpnfo.mon->PutProgressInfo(bytesread, cpnfo.len, (float)bytesread / cpnfo.len * 100.0, 1);

   if(progbar) {
      cout << endl;
   }

   if (cpnfo.len != bytesread) {
      cerr << endl << endl << 
         "File length mismatch. Read:" << bytesread << " Length:" << cpnfo.len << endl;
      retvalue = 13;
   }

#ifdef HAVE_XRDCRYPTO
   if (md5) MD_5->Final();
   if (adlerchk || md5) {
      print_chksum(src, bytesread, MD_5, adler);
   }
      
   if (summary) {        
      print_summary(src, dst, bytesread, MD_5, adler);
   }
#else
   if (adlerchk) {
      print_chksum(src, bytesread, adler);
   }
      
   if (summary) {        
      print_summary(src, dst, bytesread, adler);
   }
#endif
      
   if (retvalue >= 0) {

      for (int i = 0; i < myTIDVec.GetSize(); i++) {
         pthread_cancel(myTIDVec[i]);
         pthread_join(myTIDVec[i], &thret);	 

         delete cpnfo.XrdCli;
         cpnfo.XrdCli = 0;
      }
   }

   delete *xrddest;

   return retvalue;
}


XrdClient *BWMToken_Init(const char *bwmhost, const char *srcurl, const char *dsturl) {
   // Initialize a special client in order to get a bandwidth manager token
   // bwmhost is the hostname of the bwm to contact
   //  it can come from the one specified in the command line option -bwm
   //  it is mandatory
   //
   // src and dst are the src and dest urls, ev. 0
   //
   // The token is considered gone by the bwm server when the fake file is closed
   // or when the connection drops
   //
   if (!bwmhost[0]) return 0;

   XrdClientUrlInfo usrc(srcurl);
   XrdClientUrlInfo udst(dsturl);
   XrdOucString s = "root://";
   s += bwmhost;
   s += "//_bwm_/";
   
   s += usrc.File;

   char hname[1024];
   memset(hname, 0, sizeof(hname));

   if (gethostname(hname, sizeof(hname)))
       strcpy(hname, "Unknown");

   s += "?bwm.src=";
   if (usrc.Host != "")
      s += usrc.Host; // or the hostname() if it's local
   else
      s += hname;

   s += "?bwm.dst=";
   if (udst.Host != "")
      s += udst.Host; // or the hostname() if it's local
   else
      s += hname;

   XrdClient *cli = new XrdClient(s.c_str());
   if (cli) cli->Open(0, kXR_open_updt);
   return cli;
}

bool BWMToken_WaitFor(XrdClient *cli) {

   // Here the actual wait phase is performed through a call to kxr_query(Qvisa)
   // Note that this func is synchronous. To allow for parallel enqueueing in multiple
   // different BWMs we will have to use threads calling this func

   kXR_char buf[4096];
   // This handles the enqueueing for the current file handle opened
   if (cli) {
      if (!cli->IsOpen()) return false;
      return cli->Query(kXR_Qvisa, 0, buf, sizeof(buf));
   }
   else return true;
}


int doCp_xrd2loc(const char *src, const char *dst) {
   // ----------- xrd to loc affair
   pthread_t myTID;
   XrdClientVector<pthread_t> myTIDVec;

   void *thret;
   XrdClientStatInfo stat;
   int f;
   int retvalue = 0;

   if (BWMHost[0]) {
   // Get the queue bwm token from the local site
   XrdClient *tok1 = BWMToken_Init(BWMHost, src, dst);
   if (!tok1 || !BWMToken_WaitFor(tok1)) return 100;

   // Get the queue bwm token from the remote site
   XrdClientUrlInfo u(src);
   XrdClient *tok2 = BWMToken_Init(u.Host.c_str(), src, dst);
   if (!tok2 || !BWMToken_WaitFor(tok2)) return 100;
   }

   gettimeofday(&abs_start_time,&tz);

   // Open the input file (xrdc)
   // If Xrdcli is non-null, the correct src file has already been opened
   if (!cpnfo.XrdCli) {
      cpnfo.XrdCli = new XrdClient(src);
      if ( ( !cpnfo.XrdCli->Open(0, kXR_async) ||
	     (cpnfo.XrdCli->LastServerResp()->status != kXR_ok) ) ) {

	 cerr << "Error opening remote source file " << src << endl;
	 PrintLastServerError(cpnfo.XrdCli);

	 delete cpnfo.XrdCli;
	 cpnfo.XrdCli = 0;
	 return 1;
      }
   }

   // Open the output file (loc)
   cpnfo.XrdCli->Stat(&stat);
   cpnfo.len = stat.size;

   if (strcmp(dst, "-")) {
      // Copy to local fs
      //unlink(dst);
      f = open(dst, loc_wr_flags, 
          S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
      if (f < 0) {
	 cerr << "Error '" << strerror(errno) <<
	    "' creating " << dst << endl;

	 cpnfo.XrdCli->Close();
	 delete cpnfo.XrdCli;
	 cpnfo.XrdCli = 0;
	 return -1;
      }
      
   }
   else
      // Copy to stdout
      f = STDOUT_FILENO;


   // If the Extreme Copy flag is set, we try to find more sources for this file
   // Each source gets assigned to a different reader thread
   XrdClientVector<XrdClient *> xtremeclients;
   XrdXtRdFile *xrdxtrdfile = 0;

   if (doXtremeCp) 
      XrdXtRdFile::GetListOfSources(cpnfo.XrdCli, XtremeCpRdr, xtremeclients);

   // Start reader on xrdc
   if (doXtremeCp && (xtremeclients.GetSize() > 1)) {

      // Beware... with the extreme copy the normal read ahead mechanism
      // makes no sense at all.

      xrdxtrdfile = new XrdXtRdFile(XRDCP_BLOCKSIZE*4, cpnfo.len);

      for (int iii = 0; iii < xtremeclients.GetSize(); iii++) {
         xtreme_threadnfo *nfo = new(xtreme_threadnfo);
         nfo->xtrdhandler = xrdxtrdfile;
         nfo->cli = xtremeclients[iii];
         nfo->clientidx = xrdxtrdfile->GimmeANewClientIdx();
         nfo->startfromblk = iii*xrdxtrdfile->GetNBlks() / xtremeclients.GetSize();
         nfo->maxoutstanding = xrdmax(xrdmin( 3, xrdxtrdfile->GetNBlks() / xtremeclients.GetSize() ), 1);

         XrdSysThread::Run(&myTID, ReaderThread_xrd_xtreme, (void *)nfo);
      }

   }
   else {
      doXtremeCp = false;
      XrdSysThread::Run(&myTID, ReaderThread_xrd, (void *)&cpnfo);
   }

   int len = 1;
   void *buf;
   long long bytesread=0, offs = 0;
   long long size = cpnfo.len;
   bool draining = false;

   // Loop to write until ended or timeout err
   while (1) {

      if (xrdxtrdfile && xrdxtrdfile->AllDone()) draining = true;
      if (draining && !cpnfo.queue.GetLength()) break;

      if ( cpnfo.queue.GetBuffer(&buf, offs, len) ) {

	 if (len && buf) {

	    bytesread+=len;
	    if (progbar) {
	       gettimeofday(&abs_stop_time,&tz);
	       print_progbar(bytesread,size);
	    }

#ifdef HAVE_XRDCRYPTO
	    if (md5) {
	      MD_5->Update((const char*)buf,len);
	    }
#endif

#ifdef HAVE_LIBZ
               if (adlerchk) {
                  adler = adler32(adler, (const Bytef*)buf, len);
               }
#endif

	    if (doXtremeCp && (f != STDOUT_FILENO) && lseek(f, offs, SEEK_SET) < 0) {
	       cerr << "Error '" << strerror(errno) <<
		  "' seeking to " << dst << endl;
	       retvalue = 10;
	       break;
	    }
	    if (write(f, buf, len) <= 0) {
	       cerr << "Error '" << strerror(errno) <<
		  "' writing to " << dst << endl;
	       retvalue = 10;
	       break;
	    }

	    if (cpnfo.mon)
	      cpnfo.mon->PutProgressInfo(bytesread, cpnfo.len, (float)bytesread / cpnfo.len * 100.0);

	    free(buf);

	 }
         else
            if (!xrdxtrdfile && ( ((buf == 0) && (len == 0)) || (bytesread >= size)) ) {
               if (buf) free(buf);
               break;
            }


      }
      else {
	 cerr << endl << endl << "Critical read timeout. Unable to read data from the source." << endl;
	 retvalue = -1;
	 break;
      }
	 
      buf = 0;

   }

   if (cpnfo.mon)
     cpnfo.mon->PutProgressInfo(bytesread, cpnfo.len, (float)bytesread / cpnfo.len * 100.0, 1);

   if(progbar) {
      cout << endl;
   }

   if (cpnfo.len != bytesread) retvalue = 13;

#ifdef HAVE_XRDCRYPTO
   if (md5) MD_5->Final();
   if (md5 || adlerchk) {
      print_chksum(src, bytesread, MD_5, adler);
   }
      
   if (summary) {        
      print_summary(src,dst,bytesread,MD_5, adler);
   }      
#else
   if (adlerchk) {
      print_chksum(src, bytesread, adler);
   }
      
   if (summary) {        
      print_summary(src,dst,bytesread,adler);
   }      
#endif

   int closeres = close(f);
   if (!retvalue) retvalue = closeres;

   if (retvalue >= 0) {
      for (int i = 0; i < myTIDVec.GetSize(); i++) {
         pthread_cancel(myTIDVec[i]);
         pthread_join(myTIDVec[i], &thret);	 

         delete cpnfo.XrdCli;
         cpnfo.XrdCli = 0;
      }

      delete cpnfo.XrdCli;
      cpnfo.XrdCli = 0;
   }

   return retvalue;
}



int doCp_loc2xrd(XrdClient **xrddest, const char *src, const char * dst) {
// ----------- loc to xrd affair
   pthread_t myTID;
   void * thret;
   int retvalue = 0;
   struct stat stat;

   gettimeofday(&abs_start_time,&tz);

   // Open the input file (loc)
   cpnfo.localfile = open(src, O_RDONLY | O_BINARY);   
   if (cpnfo.localfile < 0) {
      cerr << "Error '" << strerror(errno) << "' opening " << src << endl;
      cpnfo.localfile = 0;
      return -1;
   }

   if (fstat(cpnfo.localfile, &stat)) {
     cerr << "Error '" << strerror(errno) << "' stat " << src << endl;
     cpnfo.localfile = 0;
     return -1;
   }

   // if xrddest if nonzero, then the file is already opened for writing
   if (!*xrddest) {

      *xrddest = new XrdClient(dst);
      if (!PedanticOpen4Write(*xrddest, kXR_ur | kXR_uw | kXR_gw | kXR_gr | kXR_or,
                           xrd_wr_flags) ) {
	 cerr << "Error opening remote destination file " << dst << endl;
	 PrintLastServerError(*xrddest);
         
	 close(cpnfo.localfile);
	 delete *xrddest;
	 *xrddest = 0;
	 cpnfo.localfile = 0;
	 return -1;
      }
   }
      
   // Start reader on loc
   XrdSysThread::Run(&myTID, ReaderThread_loc, (void *)&cpnfo);

   int len = 1;
   void *buf;
   long long offs = 0;
   unsigned long long bytesread=0;
   unsigned long long size = stat.st_size;
   int blkcnt = 0;

   // Loop to write until ended or timeout err
   while (len > 0) {

      if ( cpnfo.queue.GetBuffer(&buf, offs, len) ) {
	 if (len && buf) {

 	    bytesread+=len;
	    if (progbar) {
	      gettimeofday(&abs_stop_time,&tz);
	      print_progbar(bytesread,size);
	    }

#ifdef HAVE_XRDCRYPTO
	    if (md5) {
	      MD_5->Update((const char*)buf,len);
	    }
#endif

#ifdef HAVE_LIBZ
            if (adlerchk) {
               adler = adler32(adler, (const Bytef*)buf, len);
            }
#endif
	    if ( !(*xrddest)->Write(buf, offs, len) ) {
	       cerr << "Error writing to output server." << endl;
	       PrintLastServerError(*xrddest);
	       retvalue = 12;
	       break;
	    }

	    if (cpnfo.mon)
	      cpnfo.mon->PutProgressInfo(bytesread, cpnfo.len, (float)bytesread / cpnfo.len * 100.0);

	    free(buf);
	 }
	 else {
	    // If we get len == 0 then we have to stop
	    if (buf) free(buf);
	    break;
	 }
      }
      else {
	 cerr << endl << endl << "Critical read timeout. Unable to read data from the source." << endl;
	 retvalue = -1;
	 break;
      }

      buf = 0;
      blkcnt++;
   }


   if (cpnfo.mon)
     cpnfo.mon->PutProgressInfo(bytesread, cpnfo.len, (float)bytesread / cpnfo.len * 100.0, 1);

   if(progbar) {
     cout << endl;
   }

   if (size != bytesread) retvalue = 13;

#ifdef HAVE_XRDCRYPTO
   if (md5) MD_5->Final();
   if (md5 || adlerchk) {
      print_chksum(src, bytesread, MD_5, adler);
   }
   
   if (summary) {        
      print_summary(src, dst, bytesread, MD_5, adler);
   }	 
#else
   if (adlerchk) {
      print_chksum(src, bytesread, adler);
   }
   
   if (summary) {        
      print_summary(src, dst, bytesread, adler);
   }     
#endif
   
   pthread_cancel(myTID);
   pthread_join(myTID, &thret);

   delete *xrddest;
   *xrddest = 0;

   close(cpnfo.localfile);
   cpnfo.localfile = 0;

   return retvalue;
}


void PrintUsage() {
   cerr << "usage: xrdcp <source> <dest> "
     "[-d lvl] [-DSparmname stringvalue] ... [-DIparmname intvalue] [-s] [-ns] [-v]"
     " [-OS<opaque info>] [-OD<opaque info>] [-force] [-md5] [-adler] [-np] [-f] [-R] [-S]" << endl << endl;

   cerr << "<source> can be:" << endl <<
     "   a local file" << endl <<
     "   a local directory name suffixed by /" << endl <<
     "   an xrootd URL in the form root://user@host/<absolute Logical File Name in xrootd domain>" << endl <<
     "      (can be a directory. In this case the -R option can be fully honored only on a standalone server)" << endl;
   cerr << "<dest> can be:" << endl <<
     "   a local file" << endl <<
     "   a local directory name suffixed by /" << endl <<
     "   an xrootd URL in the form root://user@host/<absolute Logical File Name in xrootd domain>" << endl <<
     "      (can be a directory LFN)" << endl << endl;

   cerr << " -d lvl :         debug level: 1 (low), 2 (medium), 3 (high)" << endl;
   cerr << " -D proxyaddr:proxyport" << endl <<
           "        :         use proxyaddr:proxyport as a SOCKS4 proxy."
     " Only numerical addresses are supported." << endl <<
   cerr << " -DSparmname stringvalue" << endl <<
	   "        :         set the internal parm <parmname> with the string value <stringvalue>" << endl <<
	   "                   See XrdClientConst.hh for a list of parameters." << endl;
   cerr << " -DIparmname intvalue" << endl <<
           "        :         set the internal parm <parmname> with the integer value <intvalue>" << endl <<
           "                   See XrdClientConst.hh for a list of parameters." << endl <<
	   "                   Examples: -DIReadCacheSize 3000000 -DIReadAheadSize 1000000" << endl;
   cerr << " -s     :         silent mode, no summary output, no progress bar" << endl;
   cerr << " -np    :         no progress bar" << endl;
   cerr << " -v     :         display summary output" << endl <<endl;
   cerr << " -OS    :         adds some opaque information to any SOURCE xrootd url" << endl;
   cerr << " -OD    :         adds some opaque information to any DEST xrootd url" << endl;
   cerr << " -f     :         re-create a file if already present" << endl;
   cerr << " -F     :         set the 'force' flag for xrootd dest file opens"
     " (ignore if file is already opened)" << endl;
   cerr << " -force :         set 1-min (re)connect attempts to retry for up to 1 week,"
     " to block until xrdcp is executed" << endl << endl;
   cerr << " -md5   :         calculate the md5 checksum during transfers\n" << endl; 
#ifdef HAVE_LIBZ
   cerr << " -adler :         calculate the adler32 checksum during transfers\n" << endl; 
#endif
   cerr << " -R     :         recurse subdirectories (where it can be applied)" << endl;
   cerr << " -S num :         use <num> additional parallel streams to do the xfer." << endl << 
           "                  The max value is 15. The default is 0 (i.e. use only the main stream)" << endl;
   cerr << " -MLlibname" << endl <<
           "        :         use <libname> as external monitoring reporting library." << endl <<
           "                  The default name if XrdCpMonitorClient.so . Make sure it is reachable." << endl;
   cerr << " -X rdr :         Activate the Xtreme copy algorithm. Use 'rdr' as hostname where to query for " << endl <<
           "                  additional sources." << endl;
   cerr << " -x     :         Activate the Xtreme copy algorithm. Use the source hostname to query for " << endl <<
           "                  additional sources." << endl;
   cerr << " -P     :         request POSC (persist-on-successful-close) processing to create a new file." << endl;
   cerr << " where:" << endl;
   cerr << "   parmname     is the name of an internal parameter" << endl;
   cerr << "   stringvalue  is a string to be assigned to an internal parameter" << endl;
   cerr << "   intvalue     is an int to be assigned to an internal parameter" << endl;
}


// Main program
int main(int argc, char**argv) {
   char *srcpath = 0, *destpath = 0;
   memset (BWMHost, 0, sizeof(BWMHost));

   if (argc < 3) {
      PrintUsage();
      exit(1);
   }

#ifdef WIN32
   WORD wVersionRequested;
   WSADATA wsaData;
   int err;
   wVersionRequested = MAKEWORD( 2, 2 );
   err = WSAStartup( wVersionRequested, &wsaData );
#endif

   DebugSetLevel(-1);

   // We want this tool to be able to copy from/to everywhere
   // Note that the side effect of these calls here is to initialize the
   // XrdClient environment.
   // This is crucial if we want to later override its default values
   EnvPutString( NAME_REDIRDOMAINALLOW_RE, "*" );
   EnvPutString( NAME_CONNECTDOMAINALLOW_RE, "*" );
   EnvPutString( NAME_REDIRDOMAINDENY_RE, "" );
   EnvPutString( NAME_CONNECTDOMAINDENY_RE, "" );

   EnvPutInt( NAME_READAHEADSIZE, XRDCP_XRDRASIZE);
   EnvPutInt( NAME_READCACHESIZE, 2*XRDCP_XRDRASIZE );
   EnvPutInt( NAME_READCACHEBLKREMPOLICY, XrdClientReadCache::kRmBlk_LeastOffs );
   EnvPutInt( NAME_PURGEWRITTENBLOCKS, 1 );


   EnvPutInt( NAME_DEBUG, -1);

   for (int i=1; i < argc; i++) {

      if ( (strstr(argv[i], "-s") == argv[i])) {
	summary=false;
	progbar=false;
	continue;
      }

      if ( (strstr(argv[i], "-np") == argv[i])) {
	progbar=false;
	continue;
      }

      if ( (strstr(argv[i], "-v") == argv[i])) {
	summary=true;
	continue;
      }

      if ( (strstr(argv[i], "-R") == argv[i])) {
	recurse=true;
	continue;
      }

      if ( (strstr(argv[i], "-OS") == argv[i])) {
	 srcopaque=argv[i]+3;
	 continue;
      }
      
      if ( (strstr(argv[i], "-OD") == argv[i])) {
	 dstopaque=argv[i]+3;
	 continue;
      }

      if ( (strstr(argv[i], "-F") == argv[i])) {
	 xrd_wr_flags |= kXR_force;
	continue;
      }

      if ( (strstr(argv[i], "-P") == argv[i])) {
	 xrd_wr_flags |= kXR_posc;
	continue;
      }

      if ( (strstr(argv[i], "-f") == argv[i])) {
	// Remove the kXR_new option
	kXR_unt16 tmp = kXR_new;
	tmp = ~tmp;

	xrd_wr_flags &= tmp;

	// Also delete the existing file
	xrd_wr_flags |= kXR_delete;


	// Also set up the flags for the local fs
	loc_wr_flags = LOC_WR_FLAGS_FORCE;

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


      if ( (strstr(argv[i], "-D") == argv[i]) &&
	   (argc >= i+2) ) { 

	char host[1024];
	char *pos;
	int port;

	pos = strstr(argv[i+1], ":");

	if (pos && strlen(pos) > 1) {

	  cerr << "Using '" << argv[i+1] << "' as a SOCKS4 proxy.";
	  strncpy(host, argv[i+1], pos-argv[i+1]);
	  host[pos-argv[i+1]] = 0;

	  sscanf(pos+1, "%d", &port);

	  cerr << " Host:" << host << " port: " << port << endl;
	  EnvPutString( NAME_SOCKS4HOST, host);
	  EnvPutInt( NAME_SOCKS4PORT, port);
	}
	else {
	  cerr << "Malformed -D option." << endl;
	  exit(-1);
	}

	i++;
	continue;
      }


      if ( (strstr(argv[i], "-d") == argv[i]) &&
           (argc >= i+2) ) {
         int dbglvl = atoi(argv[i+1]);
         if (dbglvl > 0) {
            EnvPutInt( NAME_DEBUG, dbglvl);
            cerr << "Set debug level " <<  EnvGetLong(NAME_DEBUG)<< endl;
         }
         i++;
         continue;
      }

      if ( (strstr(argv[i], "-S") == argv[i]) &&
           (argc >= i+2) ) {
         int parstreams = atoi(argv[i+1]);
	 parstreams = xrdmin(parstreams, 15);
	 parstreams = xrdmax(0, parstreams);

	 EnvPutInt( NAME_MULTISTREAMCNT, parstreams);

	 cerr << "Set " << NAME_MULTISTREAMCNT << " to " <<
	   EnvGetLong(NAME_MULTISTREAMCNT) << endl;

         i++;
         continue;
      }

      if ( (strstr(argv[i], "-X") == argv[i]) &&
           (argc >= i+2) ) {
         doXtremeCp = true;
         XtremeCpRdr = argv[i+1];

	 cerr << "Extreme Copy enabled. XtremeCpRdr: " << XtremeCpRdr << endl;


         i++;
         continue;
      }

      if ( (strstr(argv[i], "-x") == argv[i]) ) {
         doXtremeCp = true;
         XtremeCpRdr = "";

	 cerr << "Extreme Copy enabled. " << endl;

         continue;
      }

#ifdef HAVE_XRDCRYPTO
#ifndef WIN32
      if ( (strstr(argv[i], "-md5") == argv[i])) {
	md5=true;

	if (!(gCryptoFactory = XrdCryptoFactory::GetCryptoFactory("ssl"))) {
	  cerr << "Cannot instantiate crypto factory ssl" << endl;
	  exit(-1);
	}

	MD_5 = gCryptoFactory->MsgDigest("md5");
	if (! MD_5) {
	  cerr << "MD object could not be instantiated " << endl;
	  exit(-1);
	}
	continue;
      }
#endif
#endif

#ifdef HAVE_LIBZ
     if ( (strstr(argv[i], "-adler") == argv[i])) {
	adlerchk=true;
	continue;
     }
#endif

      // Any other par is ignored
      if ( (strstr(argv[i], "-") == argv[i]) && (strlen(argv[i]) > 1) ) {
	 cerr << "Unknown parameter " << argv[i] << endl;
	 continue;
      }

      if (!srcpath) srcpath = argv[i];
      else
	 if (!destpath) destpath = argv[i];
      

   }

   if (!srcpath || !destpath) {
      PrintUsage();
      exit(1);
   }

   if (XtremeCpRdr == "") XtremeCpRdr = srcpath;

   DebugSetLevel(EnvGetLong(NAME_DEBUG));

   Info(XrdClientDebug::kUSERDEBUG, "main", XRDCP_VERSION);

   XrdCpWorkLst *wklst = new XrdCpWorkLst();
   XrdOucString src, dest;
   XrdClient *xrddest;

   cpnfo.XrdCli = 0;
  
   if (wklst->SetSrc(&cpnfo.XrdCli, srcpath, srcopaque, recurse)) {
     cerr << "Error accessing path/file for " << srcpath << endl;
     exit(1);
   }

   xrddest = 0;

   // From here, we will have:
   // the knowledge if the dest is a dir name or file name
   // an open instance of xrdclient if it's a file
   if (wklst->SetDest(&xrddest, destpath, dstopaque, xrd_wr_flags)) {
      cerr << "Error accessing path/file for " << destpath << endl;
      PrintLastServerError(xrddest);
      exit(1);
   }

   int retval = 0;
   while (!retval && wklst->GetCpJob(src, dest)) {
      Info(XrdClientDebug::kUSERDEBUG, "main", src << " --> " << dest);
      
#ifdef HAVE_XRDCRYPTO
      if (md5) {
	MD_5->Reset("md5");
      }
#endif
      adler = 0;


      // Initialize monitoring client, if a plugin is present
      cpnfo.mon = 0;
#ifndef WIN32
      void *monhandle = dlopen (monlibname.c_str(), RTLD_LAZY);

      if (monhandle) {
	XrdClientMonIntfHook monlibhook = (XrdClientMonIntfHook)dlsym(monhandle, "XrdClientgetMonIntf");

	const char *err = 0;
	if ((err = dlerror())) {
	  cerr << "Error accessing monitoring client library " << monhandle << ". Inappropriate content." << endl <<
	    "error: " << err << endl;
	  dlclose(monhandle);
	  monhandle = 0;
	}
	else	
	  cpnfo.mon = (XrdClientAbsMonIntf *)monlibhook(src.c_str(), dest.c_str());
      }
#endif
      
      if (cpnfo.mon) {

	char *name=0, *ver=0, *rem=0;
	if (!cpnfo.mon->GetMonLibInfo(&name, &ver, &rem)) {
	  Info(XrdClientDebug::kUSERDEBUG,
	       "main", "Monitoring client plugin found. Name:'" << name <<
	       "' Ver:'" << ver << "' Remarks:'" << rem << "'");
	}
	else {
	  delete cpnfo.mon;
	  cpnfo.mon = 0;
	}

      }

#ifndef WIN32
      if (!cpnfo.mon && monhandle) {
	dlclose(monhandle);
	monhandle = 0;
      }
#endif

      // Ok, the plugin is now loaded...
      if (cpnfo.mon) {
	// We associate the monitoring debug to the XrdClient debug level
	cpnfo.mon->Init(src.c_str(), dest.c_str(), (DebugLevel() > 0) );
	cpnfo.mon->PutProgressInfo(0, cpnfo.len, 0, 1);
      }

      if ( (src.beginswith("root://")) || (src.beginswith("xroot://")) ) {
	 // source is xrootd

	 if (srcopaque) {
	    src += "?";
	    src += srcopaque;
	 }

	 if ( (dest.beginswith("root://")) || (dest.beginswith("xroot://")) ) {
	    XrdOucString d;
	    bool isd;
	    wklst->GetDest(d, isd);

	    BuildFullDestFilename(src, d, isd);

	    if (dstopaque) {
	       d += "?";
	       d += dstopaque;
	    }

	    retval = doCp_xrd2xrd(&xrddest, src.c_str(), d.c_str());

	 }
	 else {
	    XrdOucString d;
	    bool isd;
	    int res;
	    wklst->GetDest(d, isd);
	    res = CreateDestPath_loc(d, isd);
	    if (!res || (errno == EEXIST) || !errno) {
	       BuildFullDestFilename(src, d, isd);
	       retval = doCp_xrd2loc(src.c_str(), d.c_str());
	    }
	    else
	       cerr << "Error " << strerror(errno) <<
		     " accessing path for " << d << endl;
	 }
      }
      else {
	 // source is localfs

	 if ( (dest.beginswith("root://")) || (dest.beginswith("xroot://")) ) {
	    XrdOucString d;
	    bool isd;
	    wklst->GetDest(d, isd);

	    BuildFullDestFilename(src, d, isd);

	    if (dstopaque) {
	       d += "?";
	       d += dstopaque;
	    }

	    retval = doCp_loc2xrd(&xrddest, src.c_str(), d.c_str());

	 }
	 else {
	    cerr << "Better to use cp in this case. (dest: "<<dest<<")" << endl;
	    exit(2);
	 }

      }


      if (cpnfo.mon) {
	cpnfo.mon->DeInit();
	delete cpnfo.mon;
	cpnfo.mon = 0;
#ifndef WIN32
	if (monhandle) dlclose(monhandle);
	monhandle = 0;
#endif
      }


   }

#ifdef HAVE_XRDCRYPTO
   if (md5 && MD_5) 
     delete MD_5;
#endif

   return retval;
}
