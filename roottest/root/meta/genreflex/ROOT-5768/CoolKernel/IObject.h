// $Id: IObject.h,v 1.50 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IOBJECT_H
#define COOLKERNEL_IOBJECT_H 1

// First of all, enable or disable the COOL290 API extensions (bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include <sstream>
#include "CoolKernel/ChannelId.h"
#include "CoolKernel/IRecord.h"
#include "CoolKernel/ValidityKey.h"
#include "CoolKernel/pointers.h"
#include "CoolKernel/types.h"

namespace cool
{

  // Forward declarations
  class ITime;

  /** @class IObject IObject.h
   *
   *  Abstract interface to a conditions database object.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-11-17
   */

  class IObject
  {

  public:

    /// Destructor
    virtual ~IObject() {}

    /// Instantiate a clone (deep copy) of this object
    virtual IObject* clone() const = 0;

    /// Channel identifier
    virtual const ChannelId& channelId() const = 0;

    /// Channel name
    //virtual const std::string& channelName() const = 0;

    /// Start of validity interval [since, until)
    /// NB The object is valid for since <= key < until:
    /// the object IS NOT valid for key == until
    /// For stored objects this refers to the visible validity interval
    virtual const ValidityKey& since() const = 0;

    /// End of validity interval [since, until)
    /// NB The object is valid for since <= key < until:
    /// the object IS NOT valid for key == until
    /// For stored objects this refers to the visible validity interval
    virtual const ValidityKey& until() const = 0;

    /// Data payload
    virtual const IRecord& payload() const = 0;

#ifdef COOL290VP
    /// Vector data payload
    /// There is only one iterator, payloadIterator() returns the same
    /// iterator every time it is called.
    virtual IRecordIterator& payloadIterator() const = 0;
#endif

    /// Data value of a specific payload field (as a string).
    /// Return "NULL" if the payload field is null.
    const std::string payloadValue( const std::string& name ) const
    {
      std::ostringstream o;
      payload()[name].printValue(o);
      return o.str();
    }

    /// Data value of a specific payload field (as true type).
    /// Throw FieldIsNull if the payload field is null.
    template<class T> const T& payloadValue( const std::string& name ) const
    {
      return payload()[name].template data<T>();
    }

    /// Has the object been stored into the database?
    virtual bool isStored() const = 0;

    /// System-assigned object ID
    /// (single 'surrogate PK' ID unique in the folder across all channels)
    /// Throws an exception if the object has not been stored yet
    /// WARNING! This is mainly for internal COOL developers and is subject
    /// to changes: there is no guarantee that storing the same objects
    /// will result in the same objectId values. USE AT YOUR OWN RISK!
    virtual UInt32 objectId() const = 0;

    /// System-assigned payload ID
    /// (single 'surrogate PK' ID unique in the folder across all channels)
    /// Throws an exception if the object has not been stored yet
    /// Returns 0 if the payload is not stored in a separate table
    /// WARNING! This is only for COOL developers. USE AT YOUR OWN RISK!
    virtual UInt32 payloadId() const = 0;

#ifdef COOL290VP
    /// System-assigned payload set ID
    /// Throws an exception if the object has not been stored yet
    /// Returns 0 if the payload is not stored in a separate table
    /// WARNING! This is only for COOL developers. USE AT YOUR OWN RISK!
    virtual UInt32 payloadSetId() const = 0;

    /// Payload vector size (vector folder)
    /// WARNING! This is only for COOL developers. USE AT YOUR OWN RISK!
    virtual UInt32 payloadSize() const = 0;
#endif

    /// Insertion time into the database
    /// Throws an exception if the object has not been stored yet
    virtual const ITime& insertionTime() const = 0;

    /// Start of original validity interval
    /// Throws an exception if the object has not been stored yet
    //virtual const ValidityKey& sinceOriginal() const = 0;

    /// End of original validity interval
    /// Throws an exception if the object has not been stored yet
    //virtual const ValidityKey& untilOriginal() const = 0;

    /// Insertion time of the original object into the database
    /// Throws an exception if the object has not been stored yet
    //virtual const ITime& insertionTimeOriginal() const = 0;

    /// Pretty print to an output stream
    /// The streaming operator is defined as well and using this method to
    /// allow object streaming:
    /// \code
    /// IObjectPtr obj = folder->findObject( 5 );
    /// cout << obj << endl;
    /// \endcode
    virtual std::ostream& print( std::ostream& s ) const = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IObject& operator=( const IObject& rhs );
#endif

  };

  // Define stream operator for IObjectPtr and decendants
  inline std::ostream& operator<<( std::ostream& s, const IObjectPtr& obj ) {
    return obj->print( s );
  }

  // Define stream operator for IObject and decendants
  inline std::ostream& operator<<( std::ostream& s, const IObject& obj ) {
    return obj.print( s );
  }

}

#endif // COOLKERNEL_IOBJECT_H
