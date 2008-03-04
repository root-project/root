/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndTraceEntry.cc                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonSndTraceEntry.hh"
using std::ostream;

XrdMonSndTraceEntry::XrdMonSndTraceEntry(kXR_int64 offset,
                                         kXR_int32 length,
                                         kXR_int32 id)
    : _offset(offset),
      _length(length),
      _id(id)
{}

ostream& 
operator<<(ostream& o, const XrdMonSndTraceEntry& m)
{
    o << m._offset << " " << m._length << " " << m._id;
    return o;
}




