// $Id: EventFormat_p1.cxx 294922 2013-07-24 12:41:07Z krasznaa $

// Local include(s):
#include "EventFormat_p1.h"

EventFormat_p1::EventFormat_p1()
   : m_branchNames(), m_classNames(), m_branchHashes() {

}

EventFormat_p1::EventFormat_p1( const EventFormat_p1& parent )
   : m_branchNames( parent.m_branchNames ), m_classNames( parent.m_classNames ),
     m_branchHashes( parent.m_branchHashes ) {

}

EventFormat_p1& EventFormat_p1::operator=( const EventFormat_p1& parent ) {

   m_branchNames  = parent.m_branchNames;
   m_classNames   = parent.m_classNames;
   m_branchHashes = parent.m_branchHashes;

   return *this;
}
