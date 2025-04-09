// $Id: ConstRecordAdapter.h,v 1.22 2009-12-16 17:41:24 avalassi Exp $
#ifndef RELATIONALCOOL_CONSTRECORDADAPTER_H
#define RELATIONALCOOL_CONSTRECORDADAPTER_H 1

// Include files
#include "CoolKernel/IRecord.h"
#include "CoolKernel/RecordSpecification.h"
#include "CoralBase/AttributeList.h"

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class ConstRecordAdapter ConstRecordAdapter.h
   *
   *  Read-only wrapper of a constant coral::AttributeList reference,
   *  implementing the cool::IRecord interface. The adapter can only be
   *  used as long as the AttributeList is alive. The adapter creates
   *  its own RecordSpecification from one specified at construction time.
   *
   *  All non-const methods throw: this is effectively a read-only class.
   *
   *  The StorageType constraints on the data values of all fields
   *  are enforced in all copy constructors and assignment operators.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-12-01
   */

  class ConstRecordAdapter : public IRecord {

  public:

    virtual ~ConstRecordAdapter();

    /// Constructor from a record spec and a _const_ AttributeList reference.
    /// The AttributeList must have at least all the fields required by the
    /// IRecordSpecification (with the correct name and storage type), but it
    /// may have more: it is reinterpreted according to the new specification.
    /// String attributes equal to NULL are considered equal to "".
    ConstRecordAdapter( const IRecordSpecification& spec,
                        const coral::AttributeList& al );

    /// Return the specification of this record.
    const IRecordSpecification& specification() const;

    /// Return a field in this record by its name (const).
    const IField& operator[] ( const std::string& name ) const;

    /// Return a field in this record by its index in [0, N-1] (const).
    const IField& operator[] ( UInt32 index ) const;

    /// Explicit conversion to a constant coral AttributeList reference.
    /// The AttributeList returned by this method contains only the attributes
    /// listed in the input IRecordSpecification, which may be fewer than
    /// those contained in the wrapped constant AttributeList reference.
    const coral::AttributeList& attributeList() const;

  private:

    /// Return a field in this record by its index in [0, N-1] (const).
    const IField& field( UInt32 index ) const;

    /// This method THROWS an exception because this is a read-only class.
    IField& field( UInt32 index ); // THROWS...

  private:

    /// Default constructor is private
    ConstRecordAdapter();

    /// Copy constructor is private
    ConstRecordAdapter( const ConstRecordAdapter& rhs );

    /// Assignment operator is private
    ConstRecordAdapter& operator=( const ConstRecordAdapter& rhs );

  private:

    /// The record specification.
    RecordSpecification m_spec;

    /// The record data (as an AttributeList const reference).
    /// CONST REFERENCE: the adapter can only be used as long as this is alive!
    const coral::AttributeList& m_attrList;

    /// The IField wrappers for the individual data Attribute's.
    std::vector< IField* > m_fields;

    /// The AttributeList accessible via the attributeList public method.
    coral::AttributeList m_publicAttrList;

  };

}
#endif // RELATIONALCOOL_CONSTRECORDADAPTER_H
