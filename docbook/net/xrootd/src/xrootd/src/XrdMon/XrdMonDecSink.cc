/*****************************************************************************/
/*                                                                           */
/*                             XrdMonDecSink.cc                              */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonDecSink.hh"
#include "XrdMon/XrdMonDecTraceInfo.hh"
#include "XrdMon/XrdMonDecStageInfo.hh"
#include "XrdMon/XrdMonSenderInfo.hh"
#include "XrdMon/XrdMonUtils.hh"

#include <netinet/in.h>
#include <sys/time.h> // FIXME - remove when xrootd supports openfile
#include <iomanip>
#include <unistd.h>
using std::cerr;
using std::cout;
using std::endl;
using std::ios;
using std::map;
using std::setw;

const kXR_unt16 XrdMonDecSink::VER_FREQ = 1000;

XrdMonDecSink::XrdMonDecSink(const char* baseDir,
                             const char* rtLogDir,
                             int rtBufSize,
                             bool saveTraces,
                             int maxTraceLogSize)
    : _verFreqCount(VER_FREQ),
      _rtLogger(0),
      _saveTraces(saveTraces),
      _tCacheSize(32*1024), // 32*1024 * 32 bytes = 1 MB FIXME-configurable?
      _traceLogNumber(0),
      _maxTraceLogSize(maxTraceLogSize),
      _lastSeq(0xFF),
      _uniqueDictId(1),
      _uniqueUserId(1)
{
    if ( maxTraceLogSize < 2  ) {
        cerr << "Trace log size must be > 2MB" << endl;
        throw XrdMonException(ERR_INVALIDARG, "Trace log size must be > 2MB");
    }

    _path = baseDir;
    _path += "/";
    _jnlPath = _path + "/jnl";
    _path += generateTimestamp();
    _path += "_";
    _dictPath = _path + "dict.ascii";
    _userPath = _path + "user.ascii";
    _xrdRestartLog = baseDir;
    _xrdRestartLog += "/xrdRestarts.ascii";
    
    if ( 0 == access(_dictPath.c_str(), F_OK) ) {
        string s("File "); s += _dictPath;
        s += " exists. Move it somewhere else first.";
        throw XrdMonException(ERR_INVALIDARG, s);
    }
    if ( _saveTraces ) {
        _tCache.reserve(_tCacheSize+1);
        string fTPath = _path + "trace000.ascii";
        if ( 0 == access(fTPath.c_str(), F_OK) ) {
            string s("File "); s += fTPath;
            s += " exists. Move it somewhere else first.";
            throw XrdMonException(ERR_INVALIDARG, s);
        }
    }

    if ( 0 != rtLogDir ) {
        initRT(rtLogDir, rtBufSize);
    } else {
        loadUniqueIdsAndSeq();
    }
}

XrdMonDecSink::~XrdMonDecSink()
{
    flushClosedDicts();

    reportLostPackets();
    _lost.clear();


    if ( 0 == _rtLogger ) {
        flushTCache();
        checkpoint();
    }    
    {
        XrdSysMutexHelper mh; mh.Lock(&_dMutex);
        int i, dcacheSize = _dCache.size();
        for ( i=0; i<dcacheSize ; ++i ) {
            resetDMap(i);
            delete _dCache[i];
            _dCache[i] = 0;
        }
    }
    {
        XrdSysMutexHelper mh; mh.Lock(&_uMutex);
        int i, ucacheSize = _uCache.size();
        for ( i=0; i<ucacheSize ; ++i ) {
            resetUMap(i);
            delete _uCache[i];
            _uCache[i] = 0;
        }
    }


    _rtLogger->flush();
    delete _rtLogger;

    // save ids in jnl file
    fstream f(_rtMaxIdsPath.c_str(), ios::out);
    f << "o " << _uniqueDictId << '\n'
      << "u " << _uniqueUserId << endl;
    f.close();

    // remove the flag indicating that collector/decoder is running
    unlink(_rtFlagPath.c_str());
}

struct connectDictIdsWithCache : public std::unary_function<XrdMonDecDictInfo*, void> {
    connectDictIdsWithCache(map<dictid_t, XrdMonDecDictInfo*>& dC) : _cache(dC){}
    void operator()(XrdMonDecDictInfo* di) {
        dictid_t id = di->xrdId();
        _cache[id] = di;
    }
    map<dictid_t, XrdMonDecDictInfo*>& _cache;
};

void
XrdMonDecSink::init(dictid_t min, dictid_t max, const string& senderHP)
{
    // read jnl file, create vector<XrdMonDecDictInfo*> of active 
    // XrdMonDecDictInfo objects
    vector<XrdMonDecDictInfo*> diVector = loadActiveDictInfo();

    // connect active XrdMonDecDictInfo objects to the cache
    //std::for_each(diVector.begin(),
    //              diVector.end(),
    //              connectDictIdsWithCache(_dCache));
    ::abort();
}

void
XrdMonDecSink::initRT(const char* rtLogDir,
                      int rtBufSize)
{
    _rtFlagPath = rtLogDir;
    _rtFlagPath += "/rtRunning.flag";
    _rtMaxIdsPath = rtLogDir;
    _rtMaxIdsPath += "/rtMax.jnl";

    // check if another collector/decoder is not running
    // or if the old one was closed correctly
    if( (access(_rtFlagPath.c_str(), F_OK)) != -1 ) {
        string s("Can't start rtDecoder: either collector is already running, or it has been stopped uncleanly, in which case you need to run cleanup utility first");
        throw XrdMonException(ERR_UNKNOWN, s);
    }

    // create the flag indicating that collector/decoder is running
    fstream f(_rtFlagPath.c_str(), fstream::out);
    f.close();

    char* rtLogName = new char [strlen(rtLogDir) + 32];
    sprintf(rtLogName, "%s/rtLog.txt", rtLogDir);

    char* rtLogNLock = new char [strlen(rtLogDir) + 32];
    sprintf(rtLogNLock, "%s/rtLog.lock", rtLogDir);
    _rtLogger = new XrdMonBufferedOutput(rtLogName, rtLogNLock, rtBufSize);
    addVersion();
    
    delete [] rtLogName;
    delete [] rtLogNLock;
    
    // read in unique ids from jnl file
    f.open(_rtMaxIdsPath.c_str(), ios::in);
    if ( f.is_open() ) {
        do {
            char line[64];
            f.getline(line, 64);
            char type;
            int number;
            sscanf(line, "%c %i", &type, &number);
            if ( type == 'o' && number > _uniqueDictId + 1 ) {
                _uniqueDictId = number + 1;
            }
            if ( type == 'u' && number > _uniqueUserId + 1 ) {
                _uniqueUserId = number + 1;
            }
        } while ( ! f.eof() );
        f.close();
        unlink(_rtMaxIdsPath.c_str());
        cout << "Updated uniqueIds from jnl file. "
             << "uniqueDictId: " << _uniqueDictId << ", "
             << "uniqueUserId: " << _uniqueUserId << endl;
    }
}

void
XrdMonDecSink::addDictId(dictid_t xrdId, 
                         const char* theString, 
                         int len,
                         senderid_t senderId)
{
    XrdSysMutexHelper mh; mh.Lock(&_dMutex);
    dmap_t* dMap = 0;
    if ( _dCache.size() <= senderId ) {
        dMap = new dmap_t;
        _dCache.push_back(dMap);
    } else {
        dMap = _dCache[senderId];
    }

    dmapitr_t itr = dMap->find(xrdId);
    if ( itr != dMap->end() ) {
        cerr << "Error: dictID " << xrdId << " already in cache." << endl;
        return;
        //throw XrdMonException(ERR_DICTIDINCACHE, buf);
    }
    
    XrdMonDecDictInfo* di;
    (*dMap)[xrdId] = di = new XrdMonDecDictInfo(xrdId, _uniqueDictId++, 
                                                theString, len, senderId);
    
    // cout << "Added dictInfo to sink: " << *di << endl;

    // FIXME: remove this line when xrootd supports openFile
    // struct timeval tv; gettimeofday(&tv, 0); openFile(xrdId, tv.tv_sec-8640000);
}


void
XrdMonDecSink::addStageInfo(dictid_t xrdId, 
                            const char* theString, 
                            int len,
                            senderid_t senderId)
{
    // FIXME: simplify code below once the dictId 
    // is properlysent from xrootd
    dictid_t uniqueId2use = 0;
    if ( xrdId > 0 ) {
        uniqueId2use = _uniqueDictId++;
    }

    XrdMonDecStageInfo* si;
    si = new XrdMonDecStageInfo(xrdId, uniqueId2use, 
                                theString, len, senderId);

    if ( 0 != _rtLogger ) {
        _rtLogger->add( si->writeRT2Buffer() );
        if ( --_verFreqCount < 1 ) {
            addVersion();
            _verFreqCount = VER_FREQ;
        }
    }
    delete si;
}


void
XrdMonDecSink::addUserId(dictid_t usrId, 
                         const char* theString, 
                         int len,
                         senderid_t senderId)
{
    XrdSysMutexHelper mh; mh.Lock(&_uMutex);
    umap_t* uMap = 0;
    if ( _uCache.size() <= senderId ) {
        uMap = new umap_t;
        _uCache.push_back(uMap);
    } else {
        uMap = _uCache[senderId];
    }

    umapitr_t itr = uMap->find(usrId);
    if ( itr != uMap->end() ) {
        cerr << "Error: userID " << usrId << " already in cache." << endl;
        return;
        //throw XrdMonException(ERR_USERIDINCACHE, buf);
    }
    
    XrdMonDecUserInfo* ui;
    (*uMap)[usrId] = ui 
        = new XrdMonDecUserInfo(usrId, _uniqueUserId++, 
                                theString, len, senderId);
    // cout << "Added userInfo to sink: " << *ui << endl;

    if ( 0 != _rtLogger ) {
        _rtLogger->add( ui->writeRT2Buffer(XrdMonDecUserInfo::CONNECT) );
        if ( --_verFreqCount < 1 ) {
            addVersion();
            _verFreqCount = VER_FREQ;
        }
    }
}

void
XrdMonDecSink::add(dictid_t xrdId,
                   XrdMonDecTraceInfo& trace,
                   senderid_t senderId)
{
    static long totalNoTraces = 0;
    static long noLostTraces  = 0;
    if ( ++totalNoTraces % 500001 == 500000 ) {
        cout << noLostTraces << " lost since last time" << endl;
        noLostTraces = 0;
    }

    XrdSysMutexHelper mh; mh.Lock(&_dMutex);
    dmap_t* dMap = 0;
    if ( _dCache.size() <= senderId ) {
        dMap = new dmap_t;
        _dCache.push_back(dMap);
    } else {
        dMap = _dCache[senderId];
    }
    
    dmapitr_t itr = dMap->find(xrdId);
    if ( itr == dMap->end() ) {
        registerLostPacket(xrdId, "Add trace");
        return;
    }
    XrdMonDecDictInfo* di = itr->second;
    
    trace.setUniqueId(di->uniqueId());
    
    if ( ! di->addTrace(trace) ) {
        return; // something wrong with this trace, ignore it
    }
    if ( _saveTraces ) {
        //cout << "Adding trace to sink (dictid=" 
        //<< xrdId << ") " << trace << endl;
        _tCache.push_back(trace);
        if ( _tCache.size() >= _tCacheSize ) {
            flushTCache();
        }
        cout << "FIXME: tcache one for all servers now, good enough?" << endl;
        :: abort();
    }
}

void
XrdMonDecSink::addUserDisconnect(dictid_t xrdId,
                                 kXR_int32 sec,
                                 kXR_int32 timestamp,
                                 senderid_t senderId)
{
    XrdSysMutexHelper mh; mh.Lock(&_uMutex);

    umap_t* uMap = 0;
    if ( _uCache.size() <= senderId ) {
        uMap = new umap_t;
        _uCache.push_back(uMap);
    } else {
        uMap = _uCache[senderId];
    }

    umapitr_t itr = uMap->find(xrdId);
    if ( itr == uMap->end() ) {
        registerLostPacket(xrdId, "User disconnect");
        return;
    }
    itr->second->setDisconnectInfo(sec, timestamp);
    
    if ( 0 != _rtLogger ) {
        _rtLogger->add( itr->second->writeRT2Buffer(XrdMonDecUserInfo::DISCONNECT) );
        if ( --_verFreqCount < 1 ) {
            addVersion();
            _verFreqCount = VER_FREQ;
        }
    }    
}

void
XrdMonDecSink::openFile(dictid_t xrdId,
                        kXR_int32 timestamp,
                        senderid_t senderId,
                        kXR_int64 fSize)
{
    XrdSysMutexHelper mh; mh.Lock(&_dMutex);
    dmap_t* dMap = 0;
    if ( _dCache.size() <= senderId ) {
        dMap = new dmap_t;
        _dCache.push_back(dMap);
    } else {
        dMap = _dCache[senderId];
    }
    
    dmapitr_t itr = dMap->find(xrdId);
    if ( itr == dMap->end() ) {
        registerLostPacket(xrdId, "Open file");
        cout << "requested open file " << xrdId << ", xrdId not found" << endl;
        return;
    }

    cout << "Opening file " << xrdId << endl;
    itr->second->openFile(timestamp, fSize);

    if ( 0 != _rtLogger ) {
        _rtLogger->add( itr->second->writeRT2BufferOpenFile(fSize) );
        if ( --_verFreqCount < 1 ) {
            addVersion();
            _verFreqCount = VER_FREQ;
        }
    }
}

void
XrdMonDecSink::closeFile(dictid_t xrdId, 
                         kXR_int64 bytesR, 
                         kXR_int64 bytesW, 
                         kXR_int32 timestamp,
                         senderid_t senderId)
{
    XrdSysMutexHelper mh; mh.Lock(&_dMutex);
    dmap_t* dMap = 0;
    if ( _dCache.size() <= senderId ) {
        dMap = new dmap_t;
        _dCache.push_back(dMap);
    } else {
        dMap = _dCache[senderId];
    }
    
    dmapitr_t itr = dMap->find(xrdId);
    if ( itr == dMap->end() ) {
        registerLostPacket(xrdId, "Close file");
        return;
    }

    //cout << "Closing file id= " << xrdId << " r= " 
    //     << bytesR << " w= " << bytesW << endl;
    itr->second->closeFile(bytesR, bytesW, timestamp);

    if ( 0 != _rtLogger ) {
        _rtLogger->add(itr->second->writeRT2BufferCloseFile());
        if ( --_verFreqCount < 1 ) {
            addVersion();
            _verFreqCount = VER_FREQ;
        }
    }
}

void
XrdMonDecSink::loadUniqueIdsAndSeq()
{
    if ( 0 == access(_jnlPath.c_str(), F_OK) ) {
        char buf[32];
        fstream f(_jnlPath.c_str(), ios::in);
        f.read(buf, sizeof(sequen_t)+2*sizeof(dictid_t));
        f.close();

        memcpy(&_lastSeq, buf, sizeof(sequen_t));
        kXR_int32 v32;

        memcpy(&v32, buf+sizeof(sequen_t), sizeof(kXR_int32));
        _uniqueDictId = ntohl(v32);

        memcpy(&v32, buf+sizeof(sequen_t)+sizeof(dictid_t), sizeof(kXR_int32));
        _uniqueUserId = ntohl(v32);

        cout << "Loaded from jnl file: "
             << "seq " << (int) _lastSeq
             << ", uniqueDictId " << _uniqueDictId 
             << ", uniqueUserId " << _uniqueUserId 
             << endl;
    }
}

void
XrdMonDecSink::flushClosedDicts()
{
    fstream fD(_dictPath.c_str(), ios::out | ios::app);
    enum { BUFSIZE = 1024*1024 };
    string buf;
    buf.reserve(BUFSIZE);

    int curLen = 0, sizeBefore = 0, sizeAfter = 0;
    {
        XrdSysMutexHelper mh; mh.Lock(&_dMutex);
        int i, dcacheSize = _dCache.size();
        for ( i=0; i<dcacheSize ; ++i ) {
            dmap_t* m = _dCache[i];
            sizeBefore += m->size();
            flushOneDMap(_dCache[i], curLen, BUFSIZE, buf, fD);
            sizeAfter += m->size();
        }
    }
    
    if ( curLen > 0 ) {
        fD.write(buf.c_str(), curLen);
        //cout << "flushed to disk: \n" << buf << endl;
    }
    fD.close();
    cout << "flushed (d) " << sizeBefore-sizeAfter 
         << ", left " << sizeAfter << endl;
}

void
XrdMonDecSink::flushOneDMap(dmap_t* m,
                            int& curLen,
                            const int BUFSIZE, 
                            string& buf, 
                            fstream& fD)
{
    vector<dictid_t> forDeletion;
    dmapitr_t itr;
    for ( itr=m->begin() ; itr != m->end() ; ++itr ) {
        XrdMonDecDictInfo* di = itr->second;
        if ( di != 0 && di->isClosed() ) {
            const char* dString = di->convert2string();
            int strLen = strlen(dString);
            if ( curLen == 0 ) {
                buf = dString;
            } else {
                if ( curLen + strLen >= BUFSIZE ) {
                    fD.write(buf.c_str(), curLen);
                    curLen = 0;
                    //cout << "flushed to disk: \n" << buf << endl;
                    buf = dString;
                } else {
                    buf += dString;
                }
            }
            curLen += strLen;
            delete itr->second;
            forDeletion.push_back(itr->first);
        }
    }
    int s = forDeletion.size();
    for (int i=0 ; i<s ; ++i) {
        m->erase(forDeletion[i]);
    }
}

void
XrdMonDecSink::flushUserCache()
{
    fstream fD(_userPath.c_str(), ios::app);
    enum { BUFSIZE = 1024*1024 };
    
    string buf;
    buf.reserve(BUFSIZE);

    int curLen = 0, sizeBefore = 0, sizeAfter = 0;
    {
        XrdSysMutexHelper mh; mh.Lock(&_uMutex);
        int i, ucacheSize = _uCache.size();
        for ( i=0 ; i<ucacheSize ; ++i ) {
            umap_t* m = _uCache[i];
            sizeBefore += m->size();
            flushOneUMap(m, curLen, BUFSIZE, buf, fD);
            sizeAfter += m->size();
        }        
    }
    
    if ( curLen > 0 ) {
        fD.write(buf.c_str(), curLen);
        cout << "flushed to disk: \n" << buf << endl;
    }
    fD.close();
    cout << "flushed (u) " << sizeBefore-sizeAfter << ", left " 
         << sizeAfter << endl;
}

void
XrdMonDecSink::flushOneUMap(umap_t* m,
                            int& curLen,
                            const int BUFSIZE,
                            string& buf, 
                            fstream& fD)
{
    vector <dictid_t> forDeletion;
    umapitr_t itr;
    for ( itr=m->begin() ; itr != m->end() ; ++itr ) {
        XrdMonDecUserInfo* di = itr->second;
        if ( di != 0 && di->readyToBeStored() ) {
            const char* dString = di->convert2string();
            int strLen = strlen(dString);
            if ( curLen == 0 ) {
                buf = dString;
            } else {
                if ( curLen + strLen >= BUFSIZE ) {
                    fD.write(buf.c_str(), curLen);
                    curLen = 0;
                    buf = dString;
                } else {
                    buf += dString;
                }
            }
            curLen += strLen;
            delete itr->second;
            forDeletion.push_back(itr->first);
        }
    }
    int s = forDeletion.size();
    for (int i=0 ; i<s ; ++i) {
        m->erase(forDeletion[i]);
    }
}


// used for offline processing of (full monitoring with traces) only
void
XrdMonDecSink::flushTCache()
{
    if ( _tCache.size() == 0 ) {
        return;
    }

    fstream f;
    enum { BUFSIZE = 32*1024 };    
    char buf[BUFSIZE];
    int curLen = 0;
    int s = _tCache.size();
    char oneTrace[256];
    for (int i=0 ; i<s ; ++i) {
        _tCache[i].convertToString(oneTrace);
        int strLen = strlen(oneTrace);
        if ( curLen == 0 ) {
            strcpy(buf, oneTrace);
        } else {
            if ( curLen + strLen >= BUFSIZE ) {
                write2TraceFile(f, buf, curLen);                
                curLen = 0;
                //cout << "flushed traces to disk: \n" << buf << endl;
                strcpy(buf, oneTrace);
            } else {
                strcat(buf, oneTrace);
            }
        }
        curLen += strLen;
    }
    if ( curLen > 0 ) {
        write2TraceFile(f, buf, curLen);
        //cout << "flushed traces to disk: \n" << buf << endl;
    }
    _tCache.clear();
    f.close();
}

// used for offline processing of (full monitoring with traces) only
void
XrdMonDecSink::checkpoint()
{
    ::abort();
    /*
    enum { BUFSIZE = 1024*1024 };    
    char buf[BUFSIZE];
    int bufPos = 0;
    
    // open jnl file
    fstream f(_jnlPath.c_str(), ios::out);

    // save lastSeq and uniqueIds
    memcpy(buf+bufPos, &_lastSeq, sizeof(sequen_t));
    bufPos += sizeof(sequen_t);
    kXR_int32 v = htonl(_uniqueDictId);
    memcpy(buf+bufPos, &v, sizeof(dictid_t));
    bufPos += sizeof(dictid_t);
    v = htonl(_uniqueUserId);
    memcpy(buf+bufPos, &v, sizeof(dictid_t));
    bufPos += sizeof(dictid_t);
    
    // save all active XrdMonDecDictInfos
    int nr =0;
    map<dictid_t, XrdMonDecDictInfo*>::iterator itr;
    {
        vector<dictid_t> forDeletion;
        XrdSysMutexHelper mh; mh.Lock(&_dMutex);
        for ( itr=_dCache.begin() ; itr != _dCache.end() ; ++itr ) {
            XrdMonDecDictInfo* di = itr->second;
            if ( di != 0 && ! di->isClosed() ) {
                ++nr;
                if ( di->stringSize() + bufPos >= BUFSIZE ) {
                    f.write(buf, bufPos);
                    bufPos = 0;
                }
                di->writeSelf2buf(buf, bufPos); // this will increment bufPos
                delete itr->second;
                forDeletion.push_back(itr->first);
            }
        }
        int s = forDeletion.size();
        for (int i=0 ; i<s ; ++i) {
            _dCache.erase(forDeletion[i]);
        }
    }
    if ( bufPos > 0 ) {
        f.write(buf, bufPos);
    }
    f.close();
    cout << "Saved in jnl file seq " << (int) _lastSeq
         << ", uniqueDictId " << _uniqueDictId 
         << ", uniqueUserId " << _uniqueUserId
         << " and " << nr << " XrdMonDecDictInfo objects." 
         << endl;
    */
}

// used for offline processing of (full monitoring with traces) only
void
XrdMonDecSink::openTraceFile(fstream& f)
{
    //stringstream ss(stringstream::out);
    //ss << _path << "trace"
    //   << setw(3) << setfill('0') << _traceLogNumber
    //   << ".ascii";
    //string fPath = ss.str();
    //f.open(fPath.c_str(), ios::out | ios::app);
    cout << "trace log file open NOT IMPLEMENTED " << endl;
    ::abort();
}

// used for offline processing of (full monitoring with traces) only
void
XrdMonDecSink::write2TraceFile(fstream& f, 
                               const char* buf,
                               int len)
{
    if ( ! f.is_open() ) {
        openTraceFile(f);
    }
    kXR_int64 tobeSize = len + f.tellp();
    if (  tobeSize > _maxTraceLogSize*1024*1024 ) {
        f.close();
        ++_traceLogNumber;
        openTraceFile(f);
        
    }
    f.write(buf, len);
}

vector<XrdMonDecDictInfo*>
XrdMonDecSink::loadActiveDictInfo()
{
    vector<XrdMonDecDictInfo*> v;

    if ( 0 != access(_jnlPath.c_str(), F_OK) ) {
        return v;
    }

    fstream f(_jnlPath.c_str(), ios::in);
    f.seekg(0, ios::end);
    int fSize = f.tellg();
    int pos = sizeof(sequen_t) + sizeof(kXR_int32);
    if ( fSize - pos == 0 ) {
        return v; // no active XrdMonDecDictInfo objects
    }
    f.seekg(pos); // skip seq and uniqueId
    char* buf = new char[fSize-pos];
    f.read(buf, fSize-pos);

    int bufPos = 0;
    while ( bufPos < fSize-pos ) {
        v.push_back( new XrdMonDecDictInfo(buf, bufPos) );
    }
    delete [] buf;
    
    return v;
}    

void
XrdMonDecSink::registerLostPacket(dictid_t xrdId, const char* descr)
{
    map<dictid_t, long>::iterator lostItr = _lost.find(xrdId);
    if ( lostItr == _lost.end() ) {
        cerr << descr << ": cannot find dictID " << xrdId << endl;
        _lost[xrdId] = 1;
    } else {
        ++lostItr->second;
    }
}

void
XrdMonDecSink::flushHistoryData()
{
    cout << "Flushing decoded data..." << endl;
    flushClosedDicts();
    flushUserCache();
}

void
XrdMonDecSink::reset(senderid_t senderId)
{
    flushClosedDicts();

    {
        XrdSysMutexHelper mh; mh.Lock(&_dMutex);
        resetDMap(senderId);
    }
    
    {
        XrdSysMutexHelper mh; mh.Lock(&_uMutex);
        resetUMap(senderId);
    }    
}

void
XrdMonDecSink::resetDMap(senderid_t senderId)
{
    if ( senderId >= _dCache.size() ) {
        return;
    }
    dmap_t* m = _dCache[senderId];
    dmapitr_t itr;
    for ( itr=m->begin() ; itr != m->end() ; ++itr ) {
        delete itr->second;
    }
    m->clear();
}

void
XrdMonDecSink::resetUMap(senderid_t senderId)
{
    if ( senderId >= _uCache.size() ) {
        return;
    }    
    umap_t* m = _uCache[senderId];
    umapitr_t itr;
    for ( itr=m->begin() ; itr != m->end() ; ++itr ) {
        delete itr->second;
    }
    m->clear();
}

void
XrdMonDecSink::reportLostPackets()
{
    int size = _lost.size();
    if ( size > 0 ) {
        cout << "Lost " << size << " dictIds {id, #lostTraces}: ";
        map<dictid_t, long>::iterator lostItr = _lost.begin();
        while ( lostItr != _lost.end() ) {
            cout << "{"<< lostItr->first << ", " << lostItr->second << "} ";
            ++lostItr;
        }    
        cout << endl;
    }
}

void
XrdMonDecSink::registerXrdRestart(kXR_int32 stod, senderid_t senderId)
{
    char t[24];
    timestamp2string(stod, t, GMT);
    const char* h = XrdMonSenderInfo::id2Host(senderId);

    if ( 0 != _rtLogger ) {
        char buf[512];
        sprintf(buf, "r\t%s\t%s\n", h, t);
        _rtLogger->add(buf);
    }

    fstream f(_xrdRestartLog.c_str(), ios::out | ios::app);
    f << h << '\t' << t << endl;
    f.close();
}

void
XrdMonDecSink::addVersion() 
{
    char buf[16];
    sprintf(buf, "v\t%03d\n", XRDMON_VERSION);
    _rtLogger->add(buf);
}
