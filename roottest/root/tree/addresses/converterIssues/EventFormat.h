// Dear emacs, this is -*- c++ -*-
// $Id: EventFormat.h 295134 2013-07-30 13:23:02Z krasznaa $
#ifndef EVENTFORMAT_EVENTFORMAT_H
#define EVENTFORMAT_EVENTFORMAT_H

// STL include(s):
#include <string>
#include <map>
#include <iosfwd>

// Local include(s):
#include "EventFormatElement.h"

namespace edm {

   /**
    *  @short This is the transient form of the data header information
    *
    *         The transient format provides some user friendly functions for
    *         making it easier to look up information on the fly in the analysis
    *         code.
    *
    * @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
    *
    * $Revision: 295134 $
    * $Date: 2013-07-30 15:23:02 +0200 (Tue, 30 Jul 2013) $
    */
   class EventFormat {

      /// Type of the key->data in-memory object
      typedef std::map< std::string, EventFormatElement > KeyedData_t;
      /// Type of the hash->data in-memory object
      typedef std::map< UInt_t, EventFormatElement > HashedData_t;

   public:
      /// Default constructor
      EventFormat();

      /// Add the description of a new branch
      void add( const EventFormatElement& element );

      /// Check if a description exists about a given branch
      bool exists( const std::string& key ) const;
      /// Check if a description exists about a given branch
      bool exists( UInt_t hash ) const;

      /// Get the description of a given branch
      const EventFormatElement* get( const std::string& key ) const;
      /// Get the description of a given branch
      const EventFormatElement* get( UInt_t hash ) const;

      /// Clear the object
      void clear();

      /// Iterator for looping over the elements of the object
      typedef KeyedData_t::const_iterator const_iterator;
      /// STL-like function for getting the beginning of the container
      const_iterator begin() const;
      /// STL-like function for getting the end of the container
      const_iterator end() const;

   private:
      /// Object associating string keys with the descriptions
      KeyedData_t m_keyedData; //!
      /// Object associating  hash keys with the descriptions
      HashedData_t m_hashedData; //!

   }; // class EventFormat

   /// Print operator for the event format
   std::ostream& operator<<( std::ostream& out,
                             const EventFormat& format );

} // namespace edm

// Specify a CLID for the class for Athena:
#ifndef AODX_STANDALONE
#include "CLIDSvc/CLASS_DEF.h"
CLASS_DEF( edm::EventFormat, 57337705, 1 )
#endif // AODX_STANDALONE

#endif // EVENTFORMAT_EVENTFORMAT_H
