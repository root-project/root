// $Id: IField.h,v 1.56 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IFIELD_H
#define COOLKERNEL_IFIELD_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/IFieldSpecification.h"
#include "CoolKernel/RecordException.h"

// Forward declarations
namespace coral 
{
  class Attribute;
}

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class IField IField.h
   *
   *  Abstract interface to a data field of user-defined name and storage type.
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
   *  All methods of the IField interface are const except for the two
   *  methods (setValue and setNull) that allow changes to the data values.
   *  Implementations of IField are responsible for enforcing the
   *  StorageType constraints on the data values in these two and all other
   *  non-const methods, as well as at construction and assignment time.
   *  The use of const_cast on any const method is highly discouraged
   *  as it may lead to data corruption inside the IField instance.
   *
   *  The data values at construction time are defined in the concrete
   *  implementations of the IField interface: default C++ values are
   *  recommended over nulls for all storage types (0 for numbers and
   *  chars, false for bool, "" for strings, empty blob for blobs).
   *  The value of a field is either null or a well-defined true type.
   *  Once the field has been set to null, its previous true type value
   *  is lost and the field can only be set again to non-null by setting
   *  it equal to a true type value using the setValue method: there is no
   *  such thing as a setNull(false) method to recover the previous value.
   *
   *  Note the special semantics of null values for string variables only.
   *  For strings, setNull() is now considered equivalent to setValue(""):
   *  after such calls, isNull() returns false, and data() returns "".
   *  This has been introduced to avoid inconsistencies for Oracle databases
   *  (where empty strings are treated as SQL NULL values - see bug #22381).
   *  **WARNING**: CORAL Attribute's are different from COOL Field's in the
   *  management of null values. IField::data<T>() throws if IField::isNull()
   *  returns true, while coral::Attribute::data<T>() does not throw and
   *  returns an undefined value if coral::Attribute::isNull() returns true.
   *  Note also that IField::isNull() is always false for strings.
   *
   *  It is not possible to change the name or storage types of a field
   *  via the IField interface: if necessary, this is the responsibility
   *  of the concrete classes implementing IField or IFieldSpecification.
   *
   *  Implementations of the IField interface may or may not be based on
   *  the coral::Attribute class. To simplify the port of user code to the
   *  new API, an attribute() method is provided to retrieve the contents
   *  of the IField as a (read-only) constant Attribute reference.
   *  This is DEPRECATED and may be removed in a future COOL release.
   *  **WARNING**: in line with was explained above about CORAL Attribute's
   *  being different from COOL Field's in the management of null values, the
   *  Attribute reference returned by the attribute() method will behave as
   *  follows: for all types except strings, isNull() and attribute().isNull(),
   *  as well as data() and attribute().data(), always return the same value;
   *  for strings, isNull() always returns false, while data() returns ""
   *  if either of attribute().isNull() or attribute.data<std::string>()==""
   *  are true, but it is not guaranteed that both these conditions are true.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-09-28
   */

  class IField 
  {

  public:

    virtual ~IField() {}

    /// Return the specification of this field.
    virtual const IFieldSpecification& specification() const = 0;

    /// Return the name of this field (from the spec).
    const std::string& name() const;

    /// Return the storage type of this field (from the spec).
    const StorageType& storageType() const;

    /// Is the value of this field null?
    /// For strings, this is always false.
    virtual bool isNull() const = 0;

    /// Return the data value of this field (as true type).
    /// Throw FieldWrongCppType if the field C++ type is not T.
    /// Throw FieldIsNull the field is null, i.e. if isNull() is true.
    /// For strings, return "" if setNull() was called.
    template<typename T> const T& data() const;

    /// Return the address of the true-type data value of this field.
    /// Throw FieldIsNull the field is null, i.e. if isNull() is true.
    /// For strings, return a pointer to "" if setNull() was called.
    virtual const void* addressOfData() const = 0;

    /// Set the value of this field to null: any previous value is lost.
    /// For strings, setNull() is equivalent to setValue(""): data() and
    /// addressOfData() will return "" and a pointer to "" after setNull().
    virtual void setNull() = 0;

    /// Set the value of this field to a well defined (non null) true-type.
    /// Throw FieldWrongCppType if the field C++ type is not T.
    /// For strings, setNull() is equivalent to setValue("").
    template<typename T> void setValue( const T& value );

    /// Set the value of this field equal to the value of another field
    /// (with the same StorageType, but not necessarily the same name).
    /// Throw FieldSpecificationWrongStorageType if field has the wrong type.
    void setValue( const IField& field );

    /// Compare the names, types and values of this and another field.
    virtual bool operator== ( const IField& rhs ) const;

    /// Compare the names, types and values of this and another field.
    virtual bool operator!= ( const IField& rhs ) const;

    /// Print the name, storage type and data value of this field.
    virtual std::ostream& print( std::ostream& os ) const;

    /// Print the data value of this field.
    /// Print "NULL" if the field is null (print "" for null strings).
    virtual std::ostream& printValue( std::ostream& os ) const = 0;

    /// DEPRECATED - added for easier compatibility with COOL 1.3
    /// (this is likely to be removed in a future COOL release).
    /// Explicit conversion to a constant coral Attribute reference.
    virtual const coral::Attribute& attribute() const = 0;

  private:

#ifdef COOL290CO
    /// Assignment operator is private (see bug #95823)
    IField& operator=( const IField& rhs );
#endif

    /// Compare the values of this and another field.
    /// The values of two fields are equal either if they are both non null
    /// and their true type values are equal, or if they are both null.
    /// Private method - this does not check that fields have the same type.
    virtual bool compareValue( const IField& rhs ) const = 0;

    /// Set the value of this field to a well defined (non null) true-type.
    /// For strings, setNull() is equivalent to setValue("").
    /// Throw FieldWrongCppType if the field C++ type is not cppType.
    /// Private method - this will crash if the address is not of cppType type.
    virtual void setValue( const std::type_info& cppType,
                           const void* externalAddress ) = 0;

  };

  /// Print the name, storage type and data value of a field.
  std::ostream& operator<<( std::ostream& s, const IField& field );

  //--------------------------------------------------------------------------

  inline const std::string& IField::name() const
  {
    return specification().name();
  }

  //--------------------------------------------------------------------------

  inline const StorageType& IField::storageType() const
  {
    return specification().storageType();
  }

  //--------------------------------------------------------------------------

  template<typename T>
  inline const T& IField::data() const
  {
    if ( this->storageType().cppType() != typeid(T) )
      throw FieldWrongCppType
        ( this->name(), this->storageType(), typeid(T), "IField" );
    if ( this->isNull() )
      throw FieldIsNull
        ( this->name(), this->storageType(), "IField" );
    return *( static_cast< const T* >( this->addressOfData() ) );
  }

  //--------------------------------------------------------------------------

  template<typename T>
  inline void IField::setValue( const T& value )
  {
    this->setValue( typeid(T), &value );
  }

  //--------------------------------------------------------------------------

  /// Set value to null if the other field is null, else set value from C++
  /// address of the other field's value (as in coral::Attribute::fastCopy).
  inline void IField::setValue( const IField& field )
  {
    if ( this->storageType() != field.storageType() )
      throw FieldSpecificationWrongStorageType
        ( this->name(), this->storageType(), field.storageType(), "IField" );
    if ( field.isNull() )
      this->setNull();
    else
      this->setValue( this->storageType().cppType(), field.addressOfData() );
  }

  //--------------------------------------------------------------------------

  inline bool IField::operator==( const IField& rhs ) const
  {
    // Compare names, storage types and values
    // (NB coral::Attribute::operator== does NOT compare names)
    return ( this->specification() == rhs.specification() )
      && this->compareValue( rhs );
  }

  //--------------------------------------------------------------------------

  inline bool IField::operator!=( const IField& rhs ) const
  {
    return ( ! ( *this == rhs ) );
  }

  //--------------------------------------------------------------------------

  inline std::ostream& IField::print( std::ostream& s ) const
  {
    s << name() << " (" << storageType().name() << ") : ";
    return this->printValue( s );
  }

  //--------------------------------------------------------------------------

  inline std::ostream& operator<<( std::ostream& s, const IField& field )
  {
    return field.print( s );
  }

  //--------------------------------------------------------------------------

}
#endif // COOLKERNEL_IFIELD_H
