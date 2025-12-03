// $Id: StorageType.h,v 1.19 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_STORAGETYPE_H
#define COOLKERNEL_STORAGETYPE_H 1

// Include files
#include <string>
#include <typeinfo>

namespace cool
{

  /** @class StorageType StorageType.h
   *
   *  Class describing one of the supported COOL storage types.
   *
   *  Each StorageType is associated to a unique id (enum) and a unique name.
   *
   *  Users cannot create or delete instances of this class, they can only
   *  fetch the singleton for a given id or name using static class methods.
   *
   *  A StorageType defines constraints on the allowed data values of C++
   *  variables of the cooresponding C++ type (e.g. strings associated to
   *  the String255 type must have less than 256 characters), to ensure
   *  portability across different platforms and persistent backends.
   *  While each StorageType is associated to a platform-dependent transient
   *  C++ type and to a backend-dependent persistent (e.g. SQL) data type,
   *  the StorageType class allows users to write their code to define and
   *  handles records and fields in a platform- and backend-independent way.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-11-08
   */

  class StorageType
  {

  public:

    /// Full list of the storage types supported by COOL.
    /// For each storage type, the corresponding typedef to a C++ type is
    /// included in the types.h header file (e.g. typedef bool cool::Bool).
    enum TypeId {
      Bool,
      //Char,   // not (yet?) supported
      UChar,
      Int16,
      UInt16,
      Int32,
      UInt32,
      UInt63,
      Int64,
      //UInt64, // not (yet?) supported
      Float,
      Double,
      String255,
      String4k,
      String64k,
      String16M,
      Blob64k,
      Blob16M
    };

  public:

    virtual ~StorageType() {}

    /// Return the StorageType singleton for the given type id.
    /// This method never throws because TypeId is defined as an enum
    /// (it is impossible to specify a non-existing TypeId as input argument).
    static const StorageType& storageType( const TypeId& id );

    /// Return the type id for this storage type.
    const TypeId& id() const
    {
      return m_id;
    }

    /// Return the name for this storage type.
    const std::string name() const;

    /// Return the C++ type singleton for this storage type.
    const std::type_info& cppType() const;

    /// Return the maximum allowed size for variables of this storage type
    /// (presently only applies to strings - return 0 for all other types).
    /// This limit is used internally in the validate() method, but is also
    /// needed to create the appropriate persistent types (eg SQL via CORAL).
    size_t maxSize() const;

    /// Comparison operator.
    bool operator== ( const StorageType& rhs ) const
    {
      return rhs.id() == id();
    }

    /// Comparison operator.
    bool operator!= ( const StorageType& rhs ) const
    {
      return rhs.id() != id();
    }

    /// Comparison operator using the type id.
    bool operator== ( const TypeId& rhs ) const
    {
      return rhs == id();
    }

    /// Comparison operator using the type id.
    bool operator!= ( const TypeId& rhs ) const
    {
      return rhs != id();
    }

    /// Check that a given data value is compatible with this StorageType.
    /// An optional variableName parameter may be specified to better
    /// specify the error message of any exceptions that may be thrown.
    /// Throw StorageTypeWrongCppType if the data has the wrong C++ type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    template <typename T>
    void validate( const T& data, const std::string& variableName="" ) const
    {
      validate( typeid( T ), (const void*)&data, variableName );
    }

  private:

    /// Constructor for the given type id.
    StorageType( const TypeId& id )
      : m_id( id ) {}

    // Default constructor is private and not implemented.
    StorageType();

    // Copy constructor is private and not implemented.
    StorageType( const StorageType& rhs );

    // Assignment operator is private and not implemented.
    StorageType& operator=( const StorageType& rhs );

    /// Check that a given data value is compatible with this StorageType.
    /// An optional variableName parameter may be specified to better
    /// specify the error message of any exceptions that may be thrown.
    /// Throw StorageTypeWrongCppType if the data has the wrong C++ type.
    /// Throw StorageTypeInvalidValue for values outside the allowed range.
    void validate( const std::type_info& cppTypeOfData,
                   const void* addressOfData,
                   const std::string& variableName ) const;

  private:

    /// The type id for this storage type.
    const TypeId m_id;

  };

}
#endif // COOLKERNEL_STORAGETYPE_H
