/*****************************************************************************/
/*                                                                           */
/*                             XrdMonDecSink.hh                              */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONDECSINK_HH
#define XRDMONDECSINK_HH

#include "XrdMon/XrdMonDecDictInfo.hh"
#include "XrdMon/XrdMonDecTraceInfo.hh"
#include "XrdMon/XrdMonDecUserInfo.hh"
#include "XrdMon/XrdMonDecStageInfo.hh"
#include "XrdMon/XrdMonBufferedOutput.hh"
#include "XrdSys/XrdSysPthread.hh"
#include <algorithm>
#include <fstream>
#include <map>
#include <vector>
using std::fstream;
using std::map;
using std::pair;
using std::vector;

class XrdMonDecSink {
public:
    XrdMonDecSink(const char* baseDir,
                  const char* rtLogDir,
                  int rtBufSize,
                  bool saveTraces,
                  int maxTraceLogSize);
    ~XrdMonDecSink();

    void init(dictid_t min, dictid_t max, const string& senderHP);
    sequen_t lastSeq() const { return _lastSeq; }
    void registerXrdRestart(kXR_int32 stod, senderid_t senderId);
    
    void setLastSeq(sequen_t seq) { _lastSeq = seq; }
                    
    void addDictId(dictid_t xrdId, 
                   const char* theString, 
                   int len,
                   senderid_t senderId);
    void addStageInfo(dictid_t xrdId, 
                      const char* theString, 
                      int len,
                      senderid_t senderId);
    void addUserId(dictid_t xrdId,
                   const char* theString,
                   int len,
                   senderid_t senderId);
    void add(dictid_t xrdId,
             XrdMonDecTraceInfo& trace,
             senderid_t senderId);
    void addUserDisconnect(dictid_t xrdId,
                           kXR_int32 sec,
                           kXR_int32 timestamp,
                           senderid_t senderId);
    void openFile(dictid_t dictId,
                  kXR_int32 timestamp,
                  senderid_t senderId,
                  kXR_int64 fSize);
    void closeFile(dictid_t dictId, 
                   kXR_int64 bytesR, 
                   kXR_int64 bytesW, 
                   kXR_int32 timestamp,
                   senderid_t senderId);
    void flushHistoryData();
    void flushRealTimeData() { if ( 0 != _rtLogger ) _rtLogger->flush(); }
    
    void reset(senderid_t senderId);
    
private:
    typedef map<dictid_t, XrdMonDecDictInfo*> dmap_t;
    typedef map<dictid_t, XrdMonDecUserInfo*> umap_t;
    typedef map<dictid_t, XrdMonDecDictInfo*>::iterator dmapitr_t;
    typedef map<dictid_t, XrdMonDecUserInfo*>::iterator umapitr_t;

    void initRT(const char* rtLogDir, int rtBufSize);
    void addVersion();
    
    void loadUniqueIdsAndSeq();
    vector<XrdMonDecDictInfo*> loadActiveDictInfo();
    void flushClosedDicts();
    void flushUserCache();
    void flushTCache();
    void checkpoint();
    void openTraceFile(fstream& f);
    void write2TraceFile(fstream& f, const char* buf, int len);
    void registerLostPacket(dictid_t id, const char* descr);
    void reportLostPackets();
    
    void flushOneDMap(dmap_t* m, int& curLen, const int BUFSIZE, 
                      string& buf, fstream& fD);
    void flushOneUMap(umap_t* m, int& curLen, const int BUFSIZE, 
                      string& buf, fstream& fD);

    void resetDMap(senderid_t senderId);
    void resetUMap(senderid_t senderId);
    
private:
    // this defines how frequently version information will be
    // added to the log file (every ...how many entries in the log file)
    static const kXR_unt16 VER_FREQ;

    kXR_unt16 _verFreqCount;


    vector< dmap_t* > _dCache;
    vector< umap_t* > _uCache;
    
    // The mutexes guard access to dCache, uCache respectively.
    // _dCache and _uCache can be accessed from different threads
    // (periodic data flushing inside dedicated thread)
    XrdSysMutex    _dMutex;
    XrdSysMutex    _uMutex;

    XrdMonBufferedOutput* _rtLogger;

    bool _saveTraces;
    typedef vector<XrdMonDecTraceInfo> TraceVector;
    TraceVector _tCache;
    kXR_unt32 _tCacheSize;
    kXR_unt16 _traceLogNumber;  // trace.000.ascii, 001, and so on...
    kXR_int64  _maxTraceLogSize; // [in MB]

    map<dictid_t, long> _lost; //lost dictIds -> number of lost traces
    
    sequen_t _lastSeq;
    dictid_t _uniqueDictId; // dictId in mySQL, unique for given xrootd host
    dictid_t _uniqueUserId; // userId in mySQL, unique for given xrootd host

    string _path;        // <basePath>/<date>_seqId_
    string _jnlPath;     // <basePath>/jnl
    string _dictPath;    // <basePath>/<YYYYMMDD_HH:MM:SS.MMM_dict.ascii
    string _userPath;    // <basePath>/<YYYYMMDD_HH:MM:SS.MMM_user.ascii 
    string _rtFlagPath;  // <rtLogDir>/rtRunning.flag
    string _rtMaxIdsPath;// <rtLogDir>/rtMax.jnl
    string _xrdRestartLog;// <basePath>/xrdRestarts.ascii
};

#endif /* XRDMONDECSINK_HH */
