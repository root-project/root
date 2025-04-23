// $Id: IRecordSpecification.h,v 1.31 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IRECORDSPECIFICATION_H
#define COOLKERNEL_IRECORDSPECIFICATION_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/IFieldSpecification.h"
#include "CoolKernel/types.h"

// Forward declarations
namespace coral
{
  class AttributeList;
}

namespace cool
{

  // Forward declarations
  class IRecord;

  /** @class IRecordSpecification IRecordSpecification.h
   *
   *  Abstract interface to the specification of a data record: an ordered
   *  collection of data fields of user-defined names and storage types.
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
   *  The IRecordSpecification interface only allows read-only access to
   *  the specification of a record: appropriate methods to add, remove
   *  or change the names and storage types of fields are meant to be
   *  defined and implemented in the concrete classes derived from it.
   *
   *  Field names are a property of fields and their specifications: record
   *  specifications only define which fields exist and in which order.
   *
   *  Implementations of IField and IRecord are responsible for enforcing
   *  the StorageType constraints on the data values in all of their
   *  non-const methods, as well as at construction and assignment time.
   *  The IRecordSpecification interface provides a validate() method to
   *  check if the data offered by a IRecord complies to the specification.
   *
   *  Implementations of the IRecord interface may or may not be based on
   *  the coral::AttributeList class. For internal COOL usage, another
   *  signature of the validate() method is provided to check if the data
   *  in an AttribueList complies to the record specification.
   *  This is DEPRECATED and may be removed in a future COOL release.

   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-09-22
   */

  class IRecordSpecification
  {

  public:

    virtual ~IRecordSpecification() {}

    /// Return the number of fields in this record specification.
    virtual UInt32 size() const = 0;

    /// Comparison operator. Two record specifications are equal if they have
    /// the same fields (each with the same name and type), in the same order.
    virtual bool operator==( const IRecordSpecification& rhs ) const = 0;

    /// Comparison operator. Two record specifications are equal if they have
    /// the same fields (each with the same name and type), in the same order.
    virtual bool operator!=( const IRecordSpecification& rhs ) const = 0;

    /// Does a field with this name exist?
    virtual bool exists( const std::string& name ) const = 0;

    /// Return a field given its index.
    /// Throws RecordSpecificationUnknownField if no such field exists.
    virtual const
    IFieldSpecification& operator[] ( UInt32 index ) const = 0;

    /// Return a field given its name.
    /// Throws RecordSpecificationUnknownField if no such field exists.
    virtual const
    IFieldSpecification& operator[] ( const std::string& name ) const = 0;

    /// Return the index of a field given its name.
    /// Throws RecordSpecificationUnknownField if no such field exists.
    virtual UInt32 index( const std::string& name ) const = 0;

    /// Check that a given record is compatible with this specification.
    /// For every 'reference' field in this specification, the record must
    /// contain (in any order) one field with the name, the persistent storage
    /// type and a value compatible with the storage type of that field.
    /// If checkSize is true, the record must not contain any other fields (*).
    /// Throw RecordSpecificationUnknownField if no field with a given name
    /// exists (do not check that the attribute has the same name too).
    /// Throw FieldSpecificationWrongStorageType if a field has the wrong type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    /// Throw RecordSpecificationWrongSize if record has too many fields (*).
    virtual void validate( const IRecord& record,
                           bool checkSize = true ) const = 0;

    /// DEPRECATED - added for easier compatibility with COOL 1.3
    /// (this is likely to be removed in a future COOL release).
    /// Check that an attribute list is compatible with this specification.
    /// For every 'reference' field in this specification, the list must
    /// contain (in any order) one field with the name, the transient C++ type
    /// and a value compatible with the persistent storage type of that field.
    /// If checkSize is true, the list must not contain any other fields (*).
    /// Throw coral::AttributeListException if an attribute does not exist.
    /// Throw StorageTypeWrongCppType if an attribute has the wrong C++ type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    /// Throw RecordSpecificationWrongSize if list has too many attributes (*).
    virtual void validate( const coral::AttributeList& attributeList,
                           bool checkSize = true ) const = 0;

    /*
    /// DEPRECATED - added for easier compatibility with COOL 1.3
    /// (this is likely to be removed in a future COOL release).
    /// Explicit conversion to a coral AttributeListSpecification reference.
    /// This makes it possible to construct a coral::AttributeList from
    /// a cool::IRecordSpecification, "coral::AttributeList aList( spec );".
    virtual const
    coral::AttributeListSpecification& attributeListSpecification() const = 0;
    */

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IRecordSpecification& operator=( const IRecordSpecification& rhs );
#endif

  };

}
#endif // COOLKERNEL_IRECORDSPECIFICATION_H
