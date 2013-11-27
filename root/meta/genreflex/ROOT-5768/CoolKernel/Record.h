// $Id: Record.h,v 1.38 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_RECORD_H
#define COOLKERNEL_RECORD_H 1

// Include files
#include "CoolKernel/IRecord.h"
#include "CoolKernel/RecordSpecification.h"
#include "CoralBase/AttributeList.h"

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class Record Record.h
   *
   *  Implementation of cool::IRecord interface based on the
   *  coral::AttributeList class. A Record owns its own data
   *  (stored internally as an AttributeList) and specification.
   *
   *  The StorageType constraints on the data values of all fields
   *  are enforced in all copy constructors and assignment operators.
   *  Their validation when changing these data values is delegated
   *  to the concrete implementation of IField::setValue.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-11-29
   */

  class Record : public IRecord {

  public:

    virtual ~Record();

    /// Default constructor: create a record with no fields.
    Record();

    /// Constructor: create a record with all fields in a given specification.
    /// Initialise all variables to their C++ default data values
    /// (0 for numbers and chars, false for bool, "" for strings).
    Record( const IRecordSpecification& spec );

    /// Constructor: create a record with one field of a given specification.
    /// Initialise the variable to its C++ default data value
    /// (0 for numbers and chars, false for bool, "" for strings).
    Record( const IFieldSpecification& spec );

    /// Constructor: creates a record with the given specification.
    /// Set values of the data by deep copying those in the AttributeList:
    /// this must contain at least all fields required by the specification
    /// (but it can contain more), with the correct names and C++ types and
    /// data values respecting the relevant persistent storage types.
    /// String attributes equal to NULL are considered equal to "".
    Record( const IRecordSpecification& spec, const coral::AttributeList& al );

    /// Copy constructor from another Record.
    Record( const Record& rhs );

    /// Copy constructor from any IRecord implementation.
    Record( const IRecord& rhs );

    /// Assignment operator from another Record.
    Record& operator=( const Record& rhs );

    /// Assignment operator from any other IRecord implementation.
    Record& operator=( const IRecord& rhs );

    /// Add all fields of another IRecord to both specification and data.
    void extend( const IRecord& rhs );

    /// Return the specification of this record.
    const IRecordSpecification& specification() const;

    /// Return a field in this record by its name (const).
    const IField& operator[] ( const std::string& name ) const;

    /// Return a field in this record by its name (non-const).
    IField& operator[] ( const std::string& name );

    /// Return a field in this record by its index in [0, N-1] (const).
    const IField& operator[] ( UInt32 index ) const;

    /// Return a field in this record by its index in [0, N-1] (non-const).
    IField& operator[] ( UInt32 index );

    /// Explicit conversion to a constant coral AttributeList reference.
    const coral::AttributeList& attributeList() const;

  private:

    /// Return a field in this record by its index in [0, N-1] (const).
    const IField& field( UInt32 index ) const;

    /// Return a field in this record by its index in [0, N-1] (const).
    IField& field( UInt32 index );

    /// Reset specification and data to that of an empty record with no fields.
    void reset();

  private:

    /// The record specification.
    RecordSpecification m_spec;

    /// The record data (as an AttributeList).
    coral::AttributeList m_attrList;

    /// The IField wrappers for the individual data Attribute's.
    std::vector< IField* > m_fields;

  };

}
#endif // COOLKERNEL_RECORD_H
