// $Id: IRecord.h,v 1.39 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IRECORD_H
#define COOLKERNEL_IRECORD_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include <sstream>
#include "CoolKernel/IField.h"
#include "CoolKernel/IRecordSpecification.h"

// Forward declarations
namespace coral
{
  class AttributeList;
}

namespace cool
{

  //--------------------------------------------------------------------------

  class IRecord
  {

    /** @class IRecord IRecord.h
     *
     *  Abstract interface to a data record: an ordered collection
     *  of data fields of user-defined names and storage types.
     *
     *  A field is a transient data value that can be made persistent.
     *  Its StorageType specification defines constraints on its allowed
     *  data values (e.g. strings of less than 256 characters), to ensure
     *  portability across different platforms and persistent backends.
     *  While each StorageType is associated to a platform-dependent transient
     *  C++ type and to a backend-dependent persistent (e.g. SQL) data type,
     *  the StorageType class allows users to write their code to define and
     *  handles records and fields in a platform- and backend-independent way.
     *
     *  All public methods of the IRecord interface are const: this is in
     *  practice a read-only interface. The only non-const methods are those
     *  that allow non-const access to the IField interfaces of its fields:
     *  these are protected, to be used or implemented in derived classes.
     *  Implementations of IField and IRecord are responsible for enforcing
     *  the StorageType constraints on the data values in all of their
     *  non-const methods, as well as at construction and assignment time.
     *  The use of const_cast on any const method is highly discouraged
     *  as it may lead to data corruption inside the IRecord instance.
     *
     *  It is not possible to add, remove, rename or change the types of fields
     *  via the IRecord interface: if necessary, this is the responsibility
     *  of the concrete classes implementing IRecord or IRecordSpecification.
     *
     *  Field names are a property of fields and their specifications: record
     *  specifications only define which fields exist and in which order.
     *  The IRecord base class manages field names and indexes according
     *  to its record specification. The implementation of size, index and
     *  operator[] is inlined and delegated to IRecordSpecification, while
     *  the concrete derived classes must only implement the field() method.
     *
     *  Implementations of the IRecord interface may or may not be based on
     *  the coral::AttributeList class. To simplify the port of user code,
     *  an attributeList() method is provided to retrieve the contents of
     *  the IRecord as a (read-only) constant AttributeList reference.
     *  This is DEPRECATED and may be removed in a future COOL release.
     *
     *  @author Andrea Valassi and Marco Clemencic
     *  @date   2006-09-28
     */

  public:

    virtual ~IRecord() {}

    /// Return the specification of this record.
    virtual const IRecordSpecification& specification() const = 0;

    /// Return the number of fields in this record (from the spec).
    UInt32 size() const;

    /// Return the index of a field in this record by its name (from the spec).
    UInt32 index( const std::string& name ) const;

    /// Return a field in this record by its name (const).
    const IField& operator[] ( const std::string& name ) const;

    /// Return a field in this record by its index in [0, N-1] (const).
    const IField& operator[] ( UInt32 index ) const;

    /// Comparison operator. Two records are equal if they have the same
    /// fields (each with the same name, type and value), in the same order.
    virtual bool operator== ( const IRecord& rhs ) const;

    /// Comparison operator. Two records are equal if they have the same
    /// fields (each with the same name, type and value), in the same order.
    virtual bool operator!= ( const IRecord& rhs ) const;

    /// Print the names, types and data values of all fields in this record.
    virtual std::ostream& print( std::ostream& os ) const;

    /// DEPRECATED - added for easier compatibility with COOL 1.3
    /// (this is likely to be removed in a future COOL release).
    /// Explicit conversion to a constant coral AttributeList reference.
    virtual const coral::AttributeList& attributeList() const = 0;

  protected:

    /// Return a field in this record by its name (non-const).
    IField& operator[] ( const std::string& name );

    /// Return a field in this record by its index in [0, N-1] (non-const).
    IField& operator[] ( UInt32 index );

  private:

#ifdef COOL290CO
    /// Assignment operator is private (fix bug #95823)
    IRecord& operator=( const IRecord& rhs );
#endif

    /// Return a field in this record by its index in [0, N-1] (const).
    virtual const IField& field( UInt32 index ) const = 0;

    /// Return a field in this record by its index in [0, N-1] (non-const).
    virtual IField& field( UInt32 index ) = 0;

  };

  /// Print the names, types and data values of all fields in a record.
  std::ostream& operator<<( std::ostream& s, const IRecord& record );

  //--------------------------------------------------------------------------

  inline UInt32 IRecord::size() const
  {
    return specification().size();
  }

  //--------------------------------------------------------------------------

  inline UInt32 IRecord::index( const std::string& name ) const
  {
    return specification().index( name );
  }

  //--------------------------------------------------------------------------

  inline const IField& IRecord::operator[] ( const std::string& name ) const
  {
    return field( index( name ) );
  }

  //--------------------------------------------------------------------------

  inline IField& IRecord::operator[] ( const std::string& name )
  {
    return field( index( name ) );
  }

  //--------------------------------------------------------------------------

  inline const IField& IRecord::operator[] ( UInt32 index ) const
  {
    return field( index );
  }

  //--------------------------------------------------------------------------

  inline IField& IRecord::operator[] ( UInt32 index )
  {
    return field( index );
  }

  //--------------------------------------------------------------------------

  inline bool IRecord::operator==( const IRecord& rhs ) const
  {
    if ( this->specification() != rhs.specification() ) return false;
    for ( UInt32 i = 0; i < this->size(); ++i )
      if ( (*this)[i] != rhs[i] ) return false;
    return true;
  }

  //--------------------------------------------------------------------------

  inline bool IRecord::operator!=( const IRecord& rhs ) const
  {
    return ( ! ( *this == rhs ) );
  }

  //--------------------------------------------------------------------------

  inline std::ostream& IRecord::print( std::ostream& os ) const
  {
    for ( UInt32 i = 0; i < this->size(); ++i )
    {
      if ( i > 0 ) os << ", ";
      os << "[" << (*this)[i] << "]";
    }
    return os;
  }

  //--------------------------------------------------------------------------

  inline std::ostream& operator<<( std::ostream& s, const IRecord& record )
  {
    return record.print( s );
  }

  //--------------------------------------------------------------------------

}
#endif // COOLKERNEL_IRECORD_H
