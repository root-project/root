// $Id: EventFormat.cxx 294922 2013-07-24 12:41:07Z krasznaa $

// STL include(s):
#include <iostream>
#include <iomanip>

// Local include(s):
#include "EventFormat.h"

namespace edm {

   EventFormat::EventFormat()
      : m_keyedData(), m_hashedData() {

   }

   /**
    * This function can be used to extend the object with a new element.
    *
    * @param element The element to add to the object
    */
   void EventFormat::add( const EventFormatElement& element ) {

      m_keyedData[ element.branchName() ] = element;
      m_hashedData[ element.hash() ]      = element;
      return;
   }

   /**
    * @param key The name of the branch to check
    * @returns <code>true</code> if the branch is knows,
    *          <code>false</code> if not
    */
   bool EventFormat::exists( const std::string& key ) const {

      return ( m_keyedData.find( key ) != m_keyedData.end() );
   }

   /**
    * @param hash Hashed version of the branch name to check
    * @returns <code>true</code> if the branch is knows,
    *          <code>false</code> if not
    */
   bool EventFormat::exists( UInt_t hash ) const {

      return ( m_hashedData.find( hash ) != m_hashedData.end() );
   }

   /**
    * This function can be used to get access to one element in the
    * object. Notice that the user code should first check if an element
    * exists, and only use this function if it does.
    *
    * @param key The name of the branch to get the information for
    * @returns A pointer to the element describing the requested branch
    */
   const EventFormatElement* EventFormat::get( const std::string& key ) const {

      KeyedData_t::const_iterator itr = m_keyedData.find( key );
      if( itr == m_keyedData.end() ) {
         std::cout << "<edm::TEventFormat::get>"
                   << " Information requested about unknown branch ("
                   << key << ")" << std::endl;
         return 0;
      }

      return &( itr->second );
   }

   /**
    * This function can return the element describing a given branch.
    * Notice that the user code should first check if an element
    * exists, and only use this function if it does.
    *
    * @param hash The hashed version of the name of the branch
    * @returns A pointer to the element describing the requested branch
    */
   const EventFormatElement* EventFormat::get( UInt_t hash ) const {

      HashedData_t::const_iterator itr = m_hashedData.find( hash );
      if( itr == m_hashedData.end() ) {
         std::cout << "<edm::TEventFormat::get>"
                   << " Information requested about unknown hash ("
                   << std::setw( 8 ) << std::hex << std::setfill( '0' )
                   << hash << ")" << std::endl;
         return 0;
      }

      return &( itr->second );
   }

   void EventFormat::clear() {

      m_keyedData.clear();
      m_hashedData.clear();
      return;
   }

   EventFormat::const_iterator EventFormat::begin() const {

      return m_keyedData.begin();
   }

   EventFormat::const_iterator EventFormat::end() const {

      return m_keyedData.end();
   }

   /**
    * This operator can be used for debugging purposes to print information
    * about an event format object in a user friendly way.
    *
    * @param out An output stream
    * @param format The TEventFormat to print
    * @returns The same output stream that it received
    */
   std::ostream& operator<<( std::ostream& out,
                             const EventFormat& format ) {

      out << "edm::EventFormat:";
      EventFormat::const_iterator itr = format.begin();
      EventFormat::const_iterator end = format.end();
      for( int counter = 1; itr != end; ++itr, ++counter ) {
         out << std::endl << counter << ". element: " << itr->second;
      }

      return out;
   }

} // namespace edm
