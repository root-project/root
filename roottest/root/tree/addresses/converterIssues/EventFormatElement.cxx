// $Id: EventFormatElement.cxx 294922 2013-07-24 12:41:07Z krasznaa $

// STL include(s):
#include <iostream>
#include <iomanip>

// Local include(s):
#include "EventFormatElement.h"

namespace edm {

   /**
    * @param branchName Name of the branch that is described by the object
    * @param className  Name of the transient class to be used for reading
    * @param hash       A hashed version of the branch name
    */
   EventFormatElement::EventFormatElement( const std::string& ibranchName,
                                           const std::string& iclassName,
                                           UInt_t ihash )
      : m_branchName( ibranchName ), m_className( iclassName ), m_hash( ihash ) {

   }

   /**
    * @returns The name of the branch that this object describes
    */
   const std::string& EventFormatElement::branchName() const {

      return m_branchName;
   }

   /**
    * @returns The name of the transient class to be read from the branch
    */
   const std::string& EventFormatElement::className() const {

      return m_className;
   }

   /**
    * @returns A hashed version of the branch's name
    */
   UInt_t EventFormatElement::hash() const {

      return m_hash;
   }

   /**
    * This operator can be used to print information about a given
    * event format element in a user friendly way.
    *
    * @param out An output stream
    * @param element The EventFormatElement to print
    * @returns The same output stream that it received
    */
   std::ostream& operator<<( std::ostream& out,
                             const EventFormatElement& element ) {

      out << "Branch name: " << std::setw( 30 ) << element.branchName()
          << ", Class name: " << std::setw( 30 ) << element.className()
          << ", Hash: 0x" << std::setw( 8 ) << std::hex << std::setfill( '0' )
          << element.hash() << std::setfill( ' ' );

      return out;
   }

} // namespace edm
