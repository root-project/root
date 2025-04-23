// $Id: FieldSpecification.h,v 1.22 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_FIELDSPECIFICATION_H
#define COOLKERNEL_FIELDSPECIFICATION_H 1

// Include files
#include "CoolKernel/IFieldSpecification.h"

namespace cool
{

  class FieldSpecification : public IFieldSpecification {

  public:

    virtual ~FieldSpecification();

    /// Constructor: create a field specification with the given name and type.
    /// Throw FieldSpecificationInvalidName if name is the empty string "".
    FieldSpecification( const std::string& name,
                        const StorageType::TypeId typeId );

    /// Copy constructor.
    FieldSpecification( const FieldSpecification& rhs );

    /// Return the name of this field.
    const std::string& name() const;

    /// Return the storage type of this field.
    const StorageType& storageType() const;

    /// Compare the names and storage types of this and another field.
    bool operator==( const IFieldSpecification& rhs ) const;

    /// Compare the names and storage types of this and another field.
    bool operator!=( const IFieldSpecification& rhs ) const;

    /// Check that a given field is compatible with this specification.
    /// The field must have the same transient C++ type and a value
    /// compatible with the persistent storage type of the 'reference' field.
    /// If checkName is true, the field (as well as the attribute accessible
    /// via the field) must also have the same name as the reference field (*).
    /// Throw FieldSpecificationWrongName if field/attribute name is wrong (*).
    /// Throw FieldSpecificationWrongStorageType if field has the wrong type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    void validate( const IField& field,
                   bool checkName = true ) const;

    /// Check that a given attribute is compatible with this specification.
    /// The attribute must have the same transient C++ type and a value
    /// compatible with the persistent storage type of the 'reference' field.
    /// If checkName is true, the attribute must also have the same name (*).
    /// Throw FieldSpecificationWrongName if attribute name is wrong (*).
    /// Throw StorageTypeWrongCppType if the attribute has the wrong C++ type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    void validate( const coral::Attribute& attribute,
                   bool checkName = true ) const;

  private:

    /// Default constructor is private.
    FieldSpecification();

    /// Assignment operator is private.
    FieldSpecification& operator=( const FieldSpecification& rhs );

  private:

    /// The field name.
    const std::string m_name;

    /// The field storage type.
    const StorageType& m_type;

  };

}
#endif // COOLKERNEL_FIELDSPECIFICATION_H
