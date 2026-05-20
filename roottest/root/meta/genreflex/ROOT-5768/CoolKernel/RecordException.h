// $Id: RecordException.h,v 1.18 2010-09-17 13:45:56 avalassi Exp $
#ifndef COOLKERNEL_RECORDEXCEPTION_H
#define COOLKERNEL_RECORDEXCEPTION_H 1

// Include files
#include <sstream>
#include "CoolKernel/Exception.h"
#include "CoolKernel/StorageType.h"
#include "CoolKernel/types.h"

namespace cool {

  //--------------------------------------------------------------------------

  /** @class RecordException
   *
   *  Base class for all exceptions thrown during record manipulation
   *  (by the StorageType, IField/IRecord, IField/IRecordSpecification
   *  and all related classes and subclasses).
   *
   *  @author Marco Clemencic and Andrea Valassi
   *  @date   2006-05-04
   */

  class RecordException : public Exception
  {

  public:

    /// Constructor
    explicit RecordException( const std::string& message,
                              const std::string& domain )
      : Exception( message, domain ) {}

    /// Destructor
    virtual ~RecordException() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class FieldWrongCppType
   *
   *  Exception thrown when attempting to retrieve the data of a field
   *  using the wrong C++ type.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-10
   */

  class FieldWrongCppType : public RecordException {

  public:

    /// Constructor
    explicit FieldWrongCppType( const std::string& name,
                                const StorageType& storageType,
                                const std::type_info& templateCppType,
                                const std::string& domain )
      : RecordException
    ( "Cannot get or set data for field '" + name
      + "' of storage type '" + storageType.name()
      + "' and C++ type '" + storageType.cppType().name()
      + "' using wrong C++ type '" + templateCppType.name() + "'",
      domain ) {}

    /// Destructor
    virtual ~FieldWrongCppType() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class FieldIsNull
   *
   *  Exception thrown when attempting to retrieve the data of a null field.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-10
   */

  class FieldIsNull : public RecordException {

  public:

    /// Constructor
    explicit FieldIsNull( const std::string& name,
                          const StorageType& storageType,
                          const std::string& domain )
      : RecordException
    ( "Cannot get data for field '" + name
      + "' of storage type '" + storageType.name()
      + "' and C++ type '" + storageType.cppType().name()
      + "': field is NULL",
      domain ) {}

    /// Destructor
    virtual ~FieldIsNull() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class FieldSpecificationInvalidName
   *
   *  Exception thrown when attempting to create a field specification
   *  using an invalid field name.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-09
   */

  class FieldSpecificationInvalidName : public RecordException {

  public:

    /// Constructor
    explicit FieldSpecificationInvalidName( const std::string& name,
                                            const std::string& domain )
      : RecordException
    ( "Cannot create field specification using invalid name '" + name + "'",
      domain ) {}

    /// Destructor
    virtual ~FieldSpecificationInvalidName() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class RecordSpecificationCannotExtend
   *
   *  Exception thrown when a field with a given name already exists in the
   *  relevant IRecordSpecification that the user is trying to extend.
   *
   *  @author Marco Clemencic
   *  @date   2006-05-04
   */

  class RecordSpecificationCannotExtend : public RecordException {

  public:

    /// Constructor
    explicit RecordSpecificationCannotExtend( const std::string& name,
                                              const std::string& domain )
      : RecordException
    ( "Cannot add field '" + name +
      "' to the record specification: a field with that name already exists",
      domain ) {}

    /// Destructor
    virtual ~RecordSpecificationCannotExtend() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class RecordSpecificationUnknownField
   *
   *  Exception thrown when a field with the given name or index
   *  cannot be found in the relevant record or its specification.
   *
   *  @author Andrea Valassi
   *  @date   2006-09-28
   */

  class RecordSpecificationUnknownField : public RecordException {

  public:

    /// Constructor
    explicit RecordSpecificationUnknownField( const std::string& name,
                                              const std::string& domain )
      : RecordException
    ( "Field '" + name + "' not found in the record specification", domain ) {}

    /// Constructor
    explicit RecordSpecificationUnknownField( const unsigned long index,
                                              const unsigned long size,
                                              const std::string& domain )
      : RecordException( "", domain )
    {
      std::ostringstream msg;
      msg << "Field #" << index
          << " not found in the record specification"
          << " (expected range [0," << size-1 << "])";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~RecordSpecificationUnknownField() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class RecordSpecificationWrongSize
   *
   *  Exception thrown when a record or attribute list has a different
   *  size than that required by the relevant IRecordSpecification.
   *
   *  @author Marco Clemencic and Andrea Valassi
   *  @date   2006-05-04
   */

  class RecordSpecificationWrongSize : public RecordException {

  public:

    /// Constructor
    explicit RecordSpecificationWrongSize( UInt32 expectedSize,
                                           UInt32 actualSize,
                                           const std::string& domain )
      : RecordException( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation of record or attribute list"
          << " against record specification failed:"
          << " record or attribute list has the wrong size " << actualSize
          << " (expected " << expectedSize << ")";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~RecordSpecificationWrongSize() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class FieldSpecificationWrongName
   *
   *  Exception thrown when the name of the field or attribute being validated
   *  does not match the one required by the relevant IFieldSpecification.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-06
   */

  class FieldSpecificationWrongName : public RecordException {

  public:

    /// Constructor
    explicit FieldSpecificationWrongName( const std::string& expectedName,
                                          const std::string& actualName,
                                          const std::string& domain )
      : RecordException( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation of field or attribute '" << expectedName
          << "' against field specification failed:"
          << " field or attribute has the wrong name '" << actualName << "')";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~FieldSpecificationWrongName() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class FieldSpecificationWrongStorageType
   *
   *  Exception thrown when the StorageType of the field being validated
   *  does not match the one required by the relevant IFieldSpecification.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-06
   */

  class FieldSpecificationWrongStorageType : public RecordException {

  public:

    /// Constructor
    explicit FieldSpecificationWrongStorageType
    ( const std::string& expectedName,
      const StorageType& expectedStorageType,
      const StorageType& actualStorageType,
      const std::string& domain )
      : RecordException( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation of field '" << expectedName
          << "' against StorageType '" << expectedStorageType.name()
          << "' failed: field has the wrong storage type '"
          << actualStorageType.name() << "'";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~FieldSpecificationWrongStorageType() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeException
   *
   *  Base class for all exceptions related to validation of C++ variables
   *  against the constraints defined by the relevant StorageType.
   *  These exceptions are a subset of record-related RecordException's.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-13
   */

  class StorageTypeException : public RecordException
  {

  protected:

    /// Constructor
    explicit StorageTypeException( const std::string& message,
                                   const std::string& domain )
      : RecordException( message, domain ) {}

    /// Destructor
    virtual ~StorageTypeException() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeWrongCppType
   *
   *  Exception thrown when the C++ type of the variable being validated
   *  does not match the one required by the relevant StorageType.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-13
   */

  class StorageTypeWrongCppType : public StorageTypeException {

  public:

    /// Constructor
    explicit StorageTypeWrongCppType( const std::string& expectedName,
                                      const StorageType& expectedStorageType,
                                      const std::type_info& actualCppType,
                                      const std::string& domain )
      : StorageTypeException( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation";
      if ( expectedName != "" ) msg << " of variable '" << expectedName << "'";
      msg << " against StorageType '" << expectedStorageType.name()
          << "' failed: wrong C++ type '" << actualCppType.name()
          << "' (expected '" << expectedStorageType.cppType().name() << "')";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~StorageTypeWrongCppType() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeInvalidValue
   *
   *  Base class for exceptions thrown when the value of the variable being
   *  validated breaks the constraints required by the relevant StorageType.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-13
   */

  class StorageTypeInvalidValue : public StorageTypeException {

  protected:

    /// Constructor
    explicit StorageTypeInvalidValue( const std::string& message,
                                      const std::string& domain )
      : StorageTypeException( message, domain ) {}

    /// Destructor
    virtual ~StorageTypeInvalidValue() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeInvalidUInt63
   *
   *  Base class for exceptions thrown when the value of the UInt63 variable
   *  being validated is outside the allowed range [0, 2^63-1].
   *
   *  @author Andrea Valassi
   *  @date   2006-12-14
   */

  class StorageTypeInvalidUInt63 : public StorageTypeInvalidValue {

  public:

    /// Constructor
    explicit StorageTypeInvalidUInt63( const std::string& expectedName,
                                       const UInt63& value,
                                       const std::string& domain )
      : StorageTypeInvalidValue( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation";
      if ( expectedName != "" ) msg << " of variable '" << expectedName << "'";
      msg << " against StorageType '"
          << StorageType::storageType( StorageType::UInt63 ).name()
          << "' failed: data value " << value << " is invalid"
          << " (allowed range: [" << UInt63Min << ", " << UInt63Max << "])";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~StorageTypeInvalidUInt63() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeStringTooLong
   *
   *  Exception thrown when the string variable being validated
   *  is longer than the maximum size allowed by the relevant StorageType.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-13
   */

  class StorageTypeStringTooLong : public StorageTypeInvalidValue {

  public:

    /// Constructor
    explicit StorageTypeStringTooLong
    ( const std::string& expectedName,
      const StorageType& expectedStorageType,
      size_t actualSize, // std::string::size() returns size_t
      const std::string& domain )
      : StorageTypeInvalidValue( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation";
      if ( expectedName != "" ) msg << " of variable '" << expectedName << "'";
      msg << " against StorageType '" << expectedStorageType.name()
          << "' failed: string is too long (size=" << actualSize
          << ", maxSize=" << expectedStorageType.maxSize() << ")";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~StorageTypeStringTooLong() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeStringContainsNullChar
   *
   *  Exception thrown when the string variable being validated
   *  contains the null character '\0' (or '\x00').
   *
   *  @author Andrea Valassi
   *  @date   2006-12-14
   */

  class StorageTypeStringContainsNullChar : public StorageTypeInvalidValue {

  public:

    /// Constructor
    explicit StorageTypeStringContainsNullChar
    ( const std::string& expectedName,
      const StorageType& expectedStorageType,
      const std::string& domain )
      : StorageTypeInvalidValue( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation";
      if ( expectedName != "" ) msg << " of variable '" << expectedName << "'";
      msg << " against StorageType '" << expectedStorageType.name()
          << "' failed: string contains the null character '\\0'";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~StorageTypeStringContainsNullChar() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeBlobTooLong
   *
   *  Exception thrown when the blob variable being validated
   *  is longer than the maximum size allowed by the relevant StorageType.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-16
   */

  class StorageTypeBlobTooLong : public StorageTypeInvalidValue {

  public:

    /// Constructor
    explicit StorageTypeBlobTooLong
    ( const std::string& expectedName,
      const StorageType& expectedStorageType,
      long actualSize, // coral::Blob::size() returns long
      const std::string& domain )
      : StorageTypeInvalidValue( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation";
      if ( expectedName != "" ) msg << " of variable '" << expectedName << "'";
      msg << " against StorageType '" << expectedStorageType.name()
          << "' failed: blob is too long (size=" << actualSize
          << ", maxSize=" << expectedStorageType.maxSize() << ")";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~StorageTypeBlobTooLong() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeFloatIsNaN
   *
   *  Exception thrown when the float variable being validated
   *  is Not a Number (NaN).
   *
   *  @author Andrea Valassi
   *  @date   2010-09-16
   */

  class StorageTypeFloatIsNaN : public StorageTypeInvalidValue {

  public:

    /// Constructor
    explicit StorageTypeFloatIsNaN
    ( const std::string& expectedName,
      const std::string& domain )
      : StorageTypeInvalidValue( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation";
      if ( expectedName != "" ) msg << " of variable '" << expectedName << "'";
      msg << " against StorageType '"
          << StorageType::storageType( StorageType::Float ).name()
          << "' failed: float is Not a Number (NaN)";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~StorageTypeFloatIsNaN() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class StorageTypeDoubleIsNaN
   *
   *  Exception thrown when the double variable being validated
   *  is Not a Number (NaN).
   *
   *  @author Andrea Valassi
   *  @date   2010-09-16
   */

  class StorageTypeDoubleIsNaN : public StorageTypeInvalidValue {

  public:

    /// Constructor
    explicit StorageTypeDoubleIsNaN
    ( const std::string& expectedName,
      const std::string& domain )
      : StorageTypeInvalidValue( "", domain )
    {
      std::ostringstream msg;
      msg << "Validation";
      if ( expectedName != "" ) msg << " of variable '" << expectedName << "'";
      msg << " against StorageType '"
          << StorageType::storageType( StorageType::Double ).name()
          << "' failed: double is Not a Number (NaN)";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~StorageTypeDoubleIsNaN() throw() {}

  };

  //--------------------------------------------------------------------------

}

#endif // COOLKERNEL_RECORDEXCEPTION_H
