// $Id: IFieldSpecification.h,v 1.20 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IFIELDSPECIFICATION_H
#define COOLKERNEL_IFIELDSPECIFICATION_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/StorageType.h"

// Forward declarations
namespace coral
{
  class Attribute;
}

namespace cool
{

  // Forward declarations
  class IField;

  /** @class IFieldSpecification IFieldSpecification.h
   *
   *  Abstract interface to the specification of a data field
   *  of user-defined name and storage type.
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
   *  The IFieldSpecification interface only allows read-only access to
   *  the specification of a record: appropriate methods to change the name
   *  or storage type of a field are meant to be defined and implemented in
   *  the concrete classes derived from it, if at all necessary.
   *
   *  Implementations of IField and IRecord are responsible for enforcing
   *  the StorageType constraints on the data values in all of their
   *  non-const methods, as well as at construction and assignment time.
   *  The IFieldSpecification interface provides a validate() method to
   *  check if the data offered by a IField complies to the specification.
   *
   *  Implementations of the IField interface may or may not be based
   *  on the coral::Attribute class. For internal COOL usage, another
   *  signature of the validate() method is provided to check if the
   *  data in an Attribue complies to the field specification.
   *  This is DEPRECATED and may be removed in a future COOL release.

   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-09-22
   */

  class IFieldSpecification 
  {

  public:

    virtual ~IFieldSpecification() {}

    /// Return the name of this field.
    virtual const std::string& name() const = 0;

    /// Return the storage type of this field.
    virtual const StorageType& storageType() const = 0;

    /// Compare the names and storage types of this and another field.
    virtual bool operator==( const IFieldSpecification& rhs ) const = 0;

    /// Compare the names and storage types of this and another field.
    virtual bool operator!=( const IFieldSpecification& rhs ) const = 0;

    /// Check that a given field is compatible with this specification.
    /// The field must have the same transient C++ type and a value
    /// compatible with the persistent storage type of the 'reference' field.
    /// If checkName is true, the field (as well as the attribute accessible
    /// via the field) must also have the same name as the reference field (*).
    /// Throw FieldSpecificationWrongName if field/attribute name is wrong (*).
    /// Throw FieldSpecificationWrongStorageType if field has the wrong type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    virtual void validate( const IField& field,
                           bool checkName = true ) const = 0;

    /// DEPRECATED - added for easier compatibility with COOL 1.3
    /// (this is likely to be removed in a future COOL release).
    /// Check that a given attribute is compatible with this specification.
    /// The attribute must have the same transient C++ type and a value
    /// compatible with the persistent storage type of the 'reference' field.
    /// If checkName is true, the attribute must also have the same name (*).
    /// Throw FieldSpecificationWrongName if attribute name is wrong (*).
    /// Throw StorageTypeWrongCppType if the attribute has the wrong C++ type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    virtual void validate( const coral::Attribute& attribute,
                           bool checkName = true ) const = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IFieldSpecification& operator=( const IFieldSpecification& rhs );
#endif

  };

}
#endif // COOLKERNEL_IFIELDSPECIFICATION_H
