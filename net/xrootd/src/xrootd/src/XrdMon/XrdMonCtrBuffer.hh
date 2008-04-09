/*****************************************************************************/
/*                                                                           */
/*                            XrdMonCtrBuffer.hh                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONCTRBUFFER_HH
#define XRDMONCTRBUFFER_HH

#include "XrdSys/XrdSysPthread.hh"
class XrdMonCtrPacket;

// It is a fast and simple list: elements are added 
// at the tail when they are received. 
// Archiver retrives them from the head.
// Multithreaded safe. It is a singleton.

class XrdMonCtrBuffer {
public:
    static XrdMonCtrBuffer* instance();
    void push_back(XrdMonCtrPacket* p);
    XrdMonCtrPacket* pop_front();
    void printList(const char*);
    
private:
    XrdMonCtrBuffer();
    void collectStats();

private:
    struct Elem {
        Elem(XrdMonCtrPacket* p) : packet(p), next(0) {}
        XrdMonCtrPacket* packet;
        Elem*   next;
    };

    Elem* _head;
    Elem* _tail;
    int   _noElems;
    
    XrdSysMutex    _mutex;
    XrdSysCondVar  _cond;

    // statistics
    int _max;        // maximum _noElems in the list
    int _aver;       // average noElems in the list
    int _noKInAver;  // no of elements [in 1000s] used to calculate average
    int _last1Kmax;  // max noElems in the list, last 0-1000 pop_front()
    int _last1Ktotal;// total number of noElems, last 0-1000 pop_front()
    int _counter1K;  // number of pop_front() in the "last1K"

    static XrdMonCtrBuffer* _instance;
};

#endif /* XRDMONCTRBUFFER_HH */
