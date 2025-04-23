// Dear emacs, this is -*- c++ -*-
// $Id: EventFormat_p1.h 294922 2013-07-24 12:41:07Z krasznaa $
#ifndef EVENTFORMAT_EVENTFORMAT_P1_H
#define EVENTFORMAT_EVENTFORMAT_P1_H

// STL include(s):
#include <vector>
#include <string>

/**
 *  @short Small class holding all the information needed to read the file
 *
 *         In order to be able to read the created ROOT file fully
 *         automatically, without any hard-coding that was needed for the
 *         demonstrator up to some point, one needs to save information about
 *         the layout of the file into the file. This class takes care of
 *         storing this information in an efficient way.
 *
 * @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
 *
 * $Revision: 294922 $
 * $Date: 2013-07-24 14:41:07 +0200 (Wed, 24 Jul 2013) $
 */
class EventFormat_p1 {

public:
   /// Default constructor
   EventFormat_p1();
   /// Copy constructor
   EventFormat_p1( const EventFormat_p1& parent );

   /// Assignment operator
   EventFormat_p1& operator=( const EventFormat_p1& parent );

   /// Names of the branches that we are describing
   std::vector< std::string > m_branchNames;
   /// Names of the transient objects belonging to the branch names
   std::vector< std::string > m_classNames;
   /// Hashed versions of the branch names
   std::vector< UInt_t > m_branchHashes;

}; // class EventFormat_p1

#endif // EVENTFORMAT_EVENTFORMAT_P1_H
