// $Id: RecordSpecification.h,v 1.42 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_RECORDSPECIFICATION_H
#define COOLKERNEL_RECORDSPECIFICATION_H 1

// Include files
#include <vector>
#include "CoolKernel/IRecordSpecification.h"
#include "CoolKernel/StorageType.h"

namespace cool
{

  class RecordSpecification : public IRecordSpecification {

  public:

    virtual ~RecordSpecification();

    /// Default constructor: create a record specification with no fields.
    RecordSpecification();

    /// Copy constructor from another RecordSpecification.
    RecordSpecification( const RecordSpecification& rhs );

    /// Copy constructor from any IRecordSpecification implementation.
    RecordSpecification( const IRecordSpecification& rhs );

    /// Assignment operator from another RecordSpecification.
    RecordSpecification& operator=( const RecordSpecification& rhs );

    /// Assignment operator from any IRecordSpecification implementation.
    RecordSpecification& operator=( const IRecordSpecification& rhs );

    /// Add a new field with the given name and storage type.
    /// Throw RecordSpecificationCannotExtend if a field with that name exists.
    /// Throw FieldSpecificationInvalidName if name is the empty string "".
    void extend( const std::string& name, const StorageType::TypeId typeId );

    /// Add a new field with the given name and storage type.
    /// Throw RecordSpecificationCannotExtend if a field with that name exists.
    /// Throw FieldSpecificationInvalidName if name is the empty string "".
    void extend( const std::string& name, const StorageType& type );

    /// Add a new field with the given specification.
    /// Throw RecordSpecificationCannotExtend if a field with that name exists.
    /// Throw FieldSpecificationInvalidName if name is the empty string "".
    void extend( const IFieldSpecification& fldSpec );

    /// Add all fields in the given record specification.
    /// Throw RecordSpecificationCannotExtend if a field with one name exists.
    /// Throw FieldSpecificationInvalidName if one name is the empty string "".
    void extend( const IRecordSpecification& recSpec );

    /// Return the number of fields in this record specification.
    UInt32 size() const;

    /// Comparison operator. Two record specifications are equal if they have
    /// the same fields (each with the same name and type), in the same order.
    bool operator==( const IRecordSpecification& rhs ) const;

    /// Comparison operator. Two record specifications are equal if they have
    /// the same fields (each with the same name and type), in the same order.
    bool operator!=( const IRecordSpecification& rhs ) const;

    /// Does a field with this name exist?
    bool exists( const std::string& name ) const;

    /// Return a field specification given its index in [0, N-1].
    /// Throws RecordSpecificationUnknownField if no such field exists.
    const IFieldSpecification& operator[] ( UInt32 index ) const;

    /// Return a field specification given its name.
    /// Throws RecordSpecificationUnknownField if no such field exists.
    const IFieldSpecification& operator[] ( const std::string& name ) const;

    /// Return the index of a field given its name.
    /// Throws RecordSpecificationUnknownField if no such field exists.
    UInt32 index( const std::string& name ) const;

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
    void validate( const IRecord& record,
                   bool checkSize = true ) const;

    /// Check that an attribute list is compatible with this specification.
    /// For every 'reference' field in this specification, the list must
    /// contain (in any order) one field with the name, the transient C++ type
    /// and a value compatible with the persistent storage type of that field.
    /// If checkSize is true, the list must not contain any other fields (*).
    /// Throw coral::AttributeListException if an attribute does not exist.
    /// Throw StorageTypeWrongCppType if an attribute has the wrong C++ type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    /// Throw RecordSpecificationWrongSize if list has too many attributes (*).
    void validate( const coral::AttributeList& attributeList,
                   bool checkSize = true ) const;

  private:

    /// Reset specification to that of an empty record with no fields.
    void reset();

  private:

    /// The vector of field specifications.
    std::vector< IFieldSpecification* > m_fSpecs;

  };

}
#endif // COOLKERNEL_RECORDSPECIFICATION_H
