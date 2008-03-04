/*****************************************************************************/
/*                                                                           */
/*                          XrdMonBufferedOutput.hh                          */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONBUFFEREDOUTPUT_HH
#define XRDMONBUFFEREDOUTPUT_HH

#include "XrdSys/XrdSysPthread.hh"
#include <string>
using std::string;

class XrdMonBufferedOutput {
public:
    XrdMonBufferedOutput(const char* outFileName,
                         const char* lockFileName,
                         int bufSize);
    ~XrdMonBufferedOutput();

    void add(const char* s);
    void flush(bool lockMutex=true);
    
private:
    char*       _fName;
    char*       _fNameLock;

    char*       _buf;
    const int   _bufSize; // flush when buffer is full, or when
                          // triggered by external thread
    XrdSysMutex _mutex;
};

#endif /* XRDMONBUFFEREDOUTPUT_HH */
