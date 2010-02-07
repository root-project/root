/*****************************************************************************/
/*                                                                           */
/*                            XrdMonCtrBuffer.cc                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

//         $Id$

const char *XrdMonCtrBufferCVSID = "$Id$";

#include "XrdMon/XrdMonCtrBuffer.hh"
#include "XrdMon/XrdMonCtrDebug.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <iomanip>
using std::cout;
using std::endl;
using std::setw;

XrdMonCtrBuffer* XrdMonCtrBuffer::_instance = 0;

XrdMonCtrBuffer::XrdMonCtrBuffer()
    : _head(0),
      _tail(0),
      _noElems(0),
      _max(0),
      _aver(0),
      _noKInAver(0),
      _last1Kmax(0),
      _last1Ktotal(0),
      _counter1K(1000)
{}

XrdMonCtrBuffer*
XrdMonCtrBuffer::instance() {
    if ( 0 == _instance ) {
        _instance = new XrdMonCtrBuffer();
    }
    return _instance;
}

void
XrdMonCtrBuffer::push_back(XrdMonCtrPacket* p) {
    XrdSysMutexHelper mh;
    mh.Lock(&_mutex);
    if ( 0 == _head ) {
        _tail = _head = new Elem(p);
    } else {
        _tail->next = new Elem(p);
        _tail = _tail->next;
    }

    ++_noElems;
    _cond.Signal();
}

XrdMonCtrPacket* 
XrdMonCtrBuffer::pop_front() {
    //wait until something is available...
    while ( 0 == _noElems ) {
        _cond.Wait(3600);
        if ( 0 != _head ) {
            break;
        }
    }

    collectStats();

    XrdMonCtrPacket* p = 0;
    Elem* e = 0;
    {    // retrieve
        XrdSysMutexHelper mh;
        mh.Lock(&_mutex);
        p = _head->packet;
        e = _head;
        if ( _head == _tail ) {
            _head = _tail = _head->next;
        } else {
            _head = _head->next;
        }
        --_noElems;
    }
    delete e;
    return p;
}

void
XrdMonCtrBuffer::printList(const char* txt)
{
    XrdSysMutexHelper mh; 
    mh.Lock(&XrdMonCtrDebug::_mutex);
    cout << txt << " #" << _noElems << " h" << (int *) _head << " t" << (int *) _tail << " ";
    Elem* e = _head;
    while ( e ) {
        cout << e << ":{" << (int *) e->packet << ", ->" << (int *) e->next << "} ";
        e = e->next;
    }
    cout << endl;
}

void
XrdMonCtrBuffer::collectStats()
{
    _last1Ktotal += _noElems;
    if ( _noElems > _last1Kmax ) {
        _last1Kmax = _noElems;
    }
    if ( --_counter1K == 0 ) {
        int last1Kaver = _last1Ktotal/1000;
        if ( _max < _last1Kmax ) {
            _max = _last1Kmax;
        }
        _aver = (_aver*_noKInAver + last1Kaver) / (_noKInAver+1);
        ++_noKInAver;
        { // print stats
            XrdSysMutexHelper mh;
            mh.Lock(&XrdMonCtrDebug::_mutex);
            cout << "Packet buffer stats: 1K: max "   << setw(3) << _last1Kmax
                 << ", aver "         << setw(1) << last1Kaver
                 << ". Overall: max " << setw(3) << _max
                 << ", aver "         << setw(1) << _aver
                 << ", noCalls "      << setw(3) << _noKInAver 
                 << "K" << endl;
        }
        _last1Kmax   = 0;
        _last1Ktotal = 0;
        _counter1K   = 1000;
    }
}
