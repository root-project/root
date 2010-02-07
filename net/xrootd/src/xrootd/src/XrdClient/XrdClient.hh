//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClient                                                            //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// A UNIX reference client for xrootd.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$

#ifndef XRD_CLIENT_H
#define XRD_CLIENT_H


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
// Some features:                                                       //
//  - Automatic server kind recognition (xrootd load balancer, xrootd   //
//    data server, old rootd)                                           //
//  - Fault tolerance for read/write operations (read/write timeouts    //
//    and retry)                                                        //
//  - Internal connection timeout (tunable indipendently from the OS    //
//    one)                                                              //
//  - Full implementation of the xrootd protocol                        //
//  - handling of redirections from server                              //
//  - Connection multiplexing                                           //
//  - Asynchronous operation mode                                       //
//  - High performance read caching with read-ahead                     //
//  - Thread safe                                                       //
//  - Tunable log verbosity level (0 = nothing, 3 = dump read/write     //
//    buffers too!)                                                     //
//  - Many parameters configurable. But the default are OK for nearly   //
//    all the situations.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdClient/XrdClientAbs.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdClient/XrdClientThread.hh"
#include "XrdSys/XrdSysSemWait.hh"
#include "XrdVersion.hh"


class XrdClientReadAheadMgr;

struct XrdClientOpenInfo {
    bool      inprogress;
    bool      opened;
    kXR_unt16 mode;
    kXR_unt16 options;
};

struct XrdClientStatInfo {
    int stated;
    long long size;
    long id;
    long flags;
    long modtime;
};

struct XrdClientCounters {
   int CacheSize;

   //  This does not take into account the 'suggestions'
   // like async read or async readV
   //  We count only for functions which return data, eventually
   // taken from the cache
   long long ReadBytes;
   long long WrittenBytes;
   long long WriteRequests;

   long long ReadRequests;
   long long ReadMisses;
   long long ReadHits;
   float     ReadMissRate;

   long long ReadVRequests;     // How many readVs (sync) were requested
   long long ReadVSubRequests;  // In how many sub-readVs they were split
   long long ReadVSubChunks;    // How many subchunks in total
   long long ReadVBytes;        // How many bytes were requested (sync)
       
   long long ReadVAsyncRequests;     // How many readVs (async) were requested
   long long ReadVAsyncSubRequests;  // In how many sub-readVs they were split
   long long ReadVAsyncSubChunks;    // How many subchunks in total
   long long ReadVAsyncBytes;        // How many bytes were requested (async)

   long long ReadAsyncRequests;
   long long ReadAsyncBytes;
};


class XrdClient : public XrdClientAbs {
    friend void *FileOpenerThread(void*, XrdClientThread*);
    

private:

    struct XrdClientOpenInfo    fOpenPars;   // Just a container for the last parameters
    // passed to a Open method

    // The open request can be in progress. Further requests must be delayed until
    //  finished.
    XrdSysCondVar              *fOpenProgCnd;

    // Used to open a file in parallel
    XrdClientThread            *fOpenerTh;

    // Used to limit the maximum number of concurrent opens
    static XrdSysSemWait        fConcOpenSem;

    bool                        fOpenWithRefresh;

    XrdSysCondVar              *fReadWaitData;  // Used to wait for outstanding data   

    struct XrdClientStatInfo    fStatInfo;

    long                        fReadTrimBlockSize;

    bool                        fUseCache;

    XrdOucString                fInitialUrl;
    XrdClientUrlInfo            fUrl;

    bool                        TryOpen(kXR_unt16 mode,
					kXR_unt16 options,
					bool doitparallel);

    bool                        LowOpen(const char *file,
					kXR_unt16 mode,
					kXR_unt16 options,
					char *additionalquery = 0);

    void                        TerminateOpenAttempt();

    void                        WaitForNewAsyncData();

    // Real implementation for ReadV 
    // To call it we need to be aware of the restrictions so the public
    // interface should be ReadV()
    kXR_int64                   ReadVEach(char *buf, kXR_int64 *offsets, int *lens, int &nbuf);

    bool                        IsOpenedForWrite() {
       // This supposes that no options means read only
       if (!fOpenPars.options) return false;
       
       if (fOpenPars.options & kXR_open_read) return false;
       
       return true;
    }

    XrdClientReadAheadMgr       *fReadAheadMgr;

    void                         PrintCounters();
protected:

    XrdClientCounters           fCounters;

    virtual bool                OpenFileWhenRedirected(char *newfhandle,
						       bool &wasopen);

    virtual bool                CanRedirOnError() {
      // Can redir away on error if no file is opened
      // or the file is opened in read mode

      if ( !fOpenPars.opened ) return true;

      return !IsOpenedForWrite();

    }


public:

    XrdClient(const char *url, XrdClientCallback *XrdCcb = 0, void *XrdCcbArg = 0);
    virtual ~XrdClient();

    UnsolRespProcResult         ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *sender,
						      XrdClientMessage *unsolmsg);

    bool                        Close();

    // Ask the server to flush its cache
    bool                        Sync();

    // Copy the whole file to the local filesystem. Not very efficient.
    bool                        Copy(const char *localpath);

    // Returns low level information about the cache
    bool                        GetCacheInfo(
					    // The actual cache size
					    int &size,
					    
					    // The number of bytes submitted since the beginning
					    long long &bytessubmitted,
					    
					    // The number of bytes found in the cache (estimate)
					    long long &byteshit,
					    
					    // The number of reads which did not find their data
					    // (estimate)
					    long long &misscount,
					    
					    // miss/totalreads ratio (estimate)
					    float &missrate,
					    
					    // number of read requests towards the cache
					    long long &readreqcnt,
					    
					    // ratio between bytes found / bytes submitted
					    float &bytesusefulness
       );



    // Returns client-level information about the activity performed up to now
    bool                        GetCounters( XrdClientCounters *cnt );

    // Quickly tells if the file is open
    inline bool                 IsOpen() { return fOpenPars.opened; }

    // Tells if the file opening is in progress
    bool                        IsOpen_inprogress();

    // Tells if the file is open, waiting for the completion of the parallel open
    bool                        IsOpen_wait();

    // Open the file. See the xrootd documentation for mode and options
    // If parallel, then the open is done by a separate thread, and
    // all the operations are delayed until the open has finished
    bool                        Open(kXR_unt16 mode, kXR_unt16 options, bool doitparallel=true);

    // Read a block of data. If no error occurs, it returns all the requested bytes.
    int                         Read(void *buf, long long offset, int len);

    // Read multiple blocks of data compressed into a sinle one. It's up
    // to the application to do the logistic (having the offset and len to find
    // the position of the required buffer given the big one). If no error 
    // occurs, it returns all the requested bytes.
    // NOTE: if buf == 0 then the req will be carried out asynchronously, i.e.
    // the result of the request will only populate the internal cache. A subsequent read()
    // of that chunk will get the data from the cache
    kXR_int64                   ReadV(char *buf, long long *offsets, int *lens, int nbuf);

    // Submit an asynchronous read request. Its result will only populate the cache
    //  (if any!!)
    XReqErrorType               Read_Async(long long offset, int len, bool updatecounters=true);

    // Get stat info about the file. Normally it tries to guess the file size variations
    // unless force==true
    bool                        Stat(struct XrdClientStatInfo *stinfo, bool force = false);

    // On-the-fly enabling/disabling of the cache
    bool                        UseCache(bool u = TRUE);

    // To instantly remove all the chunks in the cache
    void                        RemoveAllDataFromCache() {
        if (fConnModule)
            fConnModule->RemoveAllDataFromCache();
    }

    // To remove pieces of data from the cache
    void                        RemoveDataFromCache(long long begin_offs,
                                                    long long end_offs, bool remove_overlapped = false) {
       if (fConnModule)
          fConnModule->RemoveDataFromCache(begin_offs, end_offs, remove_overlapped);
    }

    // To set at run time the cache/readahead parameters for this instance only
    // If a parameter is < 0 then it's left untouched.
    // To simply enable/disable the caching, just use UseCache(), not this function
    void                        SetCacheParameters(int CacheSize, int ReadAheadSize, int RmPolicy);

    // To enable/disable different read ahead strategies. Defined in XrdClientReadAhead.hh
    void                        SetReadAheadStrategy(int strategy);
    
    // To enable the trimming of the blocks to read. Blocksize will be rounded to a multiple of 512.
    // Each read request will have the offset and length aligned with a multiple of blocksize
    // This strategy is similar to a read ahead, but not quite. Here we see it as a transformation
    // of the stream of the read accesses to request
    void                        SetBlockReadTrimming(int blocksize);
    
    // Truncates the open file at a specified length
    bool                        Truncate(long long len);

    // Write data to the file
    bool                        Write(const void *buf, long long offset, int len);



};

#endif
