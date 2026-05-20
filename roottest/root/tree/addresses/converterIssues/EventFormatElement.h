// Dear emacs, this is -*- c++ -*-
// $Id: EventFormatElement.h 294922 2013-07-24 12:41:07Z krasznaa $
#ifndef EVENTFORMAT_EVENTFORMATELEMENT_H
#define EVENTFORMAT_EVENTFORMATELEMENT_H

// STL include(s):
#include <string>
#include <iosfwd>

namespace edm {

   /**
    *  @short Class describing one branch of the ROOT file
    *
    *         The data header object is kept in memory in a quite different
    *         format than its persistent one. The description about single
    *         elements (branches) in the file is collected into objects
    *         like this, instead of storing this information separately in
    *         multiple STL collections.
    *
    * @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
    *
    * $Revision: 294922 $
    * $Date: 2013-07-24 14:41:07 +0200 (Wed, 24 Jul 2013) $
    */
   class EventFormatElement {

   public:
      /// Constructor with all members specified
      EventFormatElement( const std::string& branchName = "",
                          const std::string& className = "",
                          UInt_t hash = 0 );

      /// Get the branch/key name
      const std::string& branchName() const;
      /// Get the class name of this branch/key
      const std::string& className() const;
      /// Get the hash belonging to this branch/key
      UInt_t hash() const;

   private:
      /// The branch/key name
      std::string m_branchName;
      /// The class name belonging to this branch/key
      std::string m_className;
      /// The hash belonging to this branch/key
      UInt_t m_hash;

   }; // class EventFormatElement

   /// Print operator for an event format element
   std::ostream& operator<<( std::ostream& out,
                             const EventFormatElement& element );

} // namespace edm

#endif // EVENTFORMAT_EVENTFORMATELEMENT_H
