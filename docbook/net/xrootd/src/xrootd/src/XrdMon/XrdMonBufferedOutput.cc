/*****************************************************************************/
/*                                                                           */
/*                          XrdMonBufferedOutput.cc                          */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMonBufferedOutput.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include <fcntl.h>
#include <strings.h> /* bcopy */
#include <sys/stat.h>
#include <fstream>
#include <stdio.h>
using std::cout;
using std::endl;
using std::fstream;
using std::ios;


XrdMonBufferedOutput::XrdMonBufferedOutput(const char* outFileName,
                                           const char* lockFileName,
                                           int bufSize)
                                       
    : _buf(0), 
      _bufSize(bufSize)
{
    _fName = new char[strlen(outFileName)+1];
    strcpy(_fName, outFileName);

    if ( lockFileName == 0 ) {
        _fNameLock = new char [strlen(outFileName)+8];
        sprintf(_fNameLock, "%s.lock", outFileName);
    } else {
        _fNameLock = new char [strlen(lockFileName)+1];
        sprintf(_fNameLock, lockFileName);
    }
    
    _buf = new char [_bufSize];
    strcpy(_buf, "");
}

XrdMonBufferedOutput::~XrdMonBufferedOutput()
{
    delete [] _fName;
    delete [] _fNameLock;
    delete [] _buf;
}

void
XrdMonBufferedOutput::add(const char* s)
{
    XrdSysMutexHelper mh; mh.Lock(&_mutex);

    if ( static_cast<int>(strlen(_buf) + strlen(s)) >= _bufSize ) {
        flush(false); // false -> don't lock mutex, already locked
    }
    strcat(_buf, s);
}

void
XrdMonBufferedOutput::flush(bool lockIt)
{
    // get the lock, wait if necessary
    struct flock lock_args;
    bzero(&lock_args, sizeof(lock_args));

    mode_t m = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;

    cout << "RT locking." << std::flush;
    int fLock = open(_fNameLock, O_WRONLY|O_CREAT, m);
    lock_args.l_type = F_WRLCK;
    fcntl(fLock, F_SETLKW, &lock_args);    
    cout << "ok." << std::flush;

    int s = strlen(_buf);
    if ( s > 0 ) {        
        XrdSysMutexHelper mh;
        if ( lockIt ) {
            mh.Lock(&_mutex);
        }
        // open rt log, write to it, and close it
        int f = open(_fName, O_WRONLY|O_CREAT|O_APPEND,m);
        write(f, _buf, strlen(_buf));
        close(f);
        // clear buffer
        strcpy(_buf, "");
    }
    cout << s;

    // unlock
    bzero(&lock_args, sizeof(lock_args));
    lock_args.l_type = F_UNLCK;
    fcntl(fLock, F_SETLKW, &lock_args);
    close (fLock);
    
    cout << ".unlocked" << endl;
}

