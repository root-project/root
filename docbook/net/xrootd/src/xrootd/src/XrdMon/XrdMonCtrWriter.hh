/*****************************************************************************/
/*                                                                           */
/*                            XrdMonCtrWriter.hh                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONCTRWRITER_HH
#define XRDMONCTRWRITER_HH

#include "XrdMon/XrdMonTypes.hh"
#include <fstream>
#include <string>
using std::fstream;
using std::ostream;
using std::string;

class XrdMonHeader;

// Class writes data to a log file.
// One instance per one xrootd instance.
// It buffers data in memory to avoid 
// overloading disks

class XrdMonCtrWriter {
public:
    XrdMonCtrWriter(senderid_t senderId, kXR_int32 stod);
    ~XrdMonCtrWriter();
    void operator()(const char* packet, 
                    const XrdMonHeader& header, 
                    long currentTime);

    kXR_int32 prevStod() const { return _prevStod; }

    void forceClose();
    long lastActivity() const { return _lastActivity; }

    static void setBaseDir(const char* dir) { _baseDir    = dir; }
    static void setMaxLogSize(kXR_int64 size) { _maxLogSize = size;}
    static void setBufferSize(int size)     { _bufferSize = size;}
    
private:
    enum LogType { ACTIVE, PERMANENT };

    bool logIsOpen() { return _file.is_open(); }
    bool logIsFull() { return (kXR_int64) _file.tellp() >= _maxLogSize; }
    bool bufferIsFull(packetlen_t x) { return _bPos + x > _bufferSize; }

    string logName(LogType t) const;
    void mkActiveLogNameDirs() const;
    
    void flushBuffer();
    void openLog();
    void closeLog();
    void publish();

private:
    static string  _baseDir;
    static kXR_int64 _maxLogSize;
    static int     _bufferSize;
    static long    _totalArchived;

    kXR_int32 _prevStod; // for checking if xrootd restarted
    
    string  _timestamp;
    hp_t    _sender;     // { hostName, portNr }
    char*   _buffer;
    kXR_int32 _bPos;     // position where to write to buffer
    fstream _file;       // non-published log file

    long _lastActivity; // approx time of last activity

    friend ostream& operator<<(ostream& o, const XrdMonCtrWriter& w);
};

#endif /* XRDMONCTRWRITER_HH */
