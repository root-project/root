/*****************************************************************************/
/*                                                                           */
/*                           XrdMonCout2FileApp.cc                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$


// Reads from standard input, buffers incoming data and flushes to file
// every x sec or whenever buffer is full whichever comes first (x is set
// via argument). It locks the lock file before doing any io.

#include "XrdMon/XrdMonBufferedOutput.hh"
#include "XProtocol/XPtypes.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include <unistd.h> /* access */
#include <assert.h>
using namespace std;

int flushFreq = 30; // in seconds

extern "C"
void* flush2disk(void* arg)
{
    XrdMonBufferedOutput* bOut = (XrdMonBufferedOutput*) arg;
    assert ( 0 != bOut );

    while ( 1 ) {
        sleep(flushFreq);
        bOut->flush();
    }
    return (void*) 0;
}

int main(int argc, char* argv[])
{
    if ( argc != 2 ) {
        cerr << "Expected arg: <outFileName>" << endl;
        return 1;
    }

    const char* fName = argv[1];
    const kXR_int32 bufSize = 128*1024; // Make it configurable?

    XrdMonBufferedOutput bOut(fName, (const char*) 0, bufSize);

    pthread_t flushThread;
    if ( 0 != pthread_create(&flushThread,
                             0,
                             flush2disk,
                             (void*)&bOut) ) {
        cerr << "Can't start flush2disk thread" << endl;
        return 2;
    }

    char line[1024];
    while ( 1 ) {
        cin.getline(line, 1024);
        strcat(line, "\n");
        bOut.add(line);
    }

    // never reached    
    bOut.flush();
    return 0;
}


