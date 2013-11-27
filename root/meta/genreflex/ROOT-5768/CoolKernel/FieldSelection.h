// $Id: FieldSelection.h,v 1.16 2009-12-17 18:50:42 avalassi Exp $
#ifndef COOLKERNEL_FIELDSELECTION_H
#define COOLKERNEL_FIELDSELECTION_H 1

// Include files
#include "CoolKernel/FieldSpecification.h"
#include "CoolKernel/IRecordSelection.h"
#include "CoolKernel/Record.h"

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class FieldSelection FieldSelection.h
   *
   *  Simple selection on a data field.
   *
   *  @author Andrea Valassi and Martin Wache
   *  @date   2008-07-28
   */

  class FieldSelection : virtual public IRecordSelection
  {

  public:

    /// Binary relation operator (comparison to a reference value).
    enum Relation { EQ, NE, GT, GE, LT, LE };

    /// Unary nullness operator (comparison to NULL).
    enum Nullness { IS_NULL, IS_NOT_NULL };

    /// Describe a binary relation operator.
    static const std::string describe( Relation relation );

    /// Describe a unary nullness operator.
    static const std::string describe( Nullness nullness );

  public:

    /// Destructor.
    virtual ~FieldSelection();

    /// Constructor for comparison to non-NULL reference values.
    template<typename T>
    FieldSelection( const std::string& name,
                    const StorageType::TypeId typeId,
                    Relation relation,
                    const T& refValue );

    /// Constructor for comparison to NULL.
    FieldSelection( const std::string& name,
                    const StorageType::TypeId typeId,
                    Nullness nullness );

    /// Can the selection be applied to a record with the given specification?
    bool canSelect( const IRecordSpecification& spec ) const;

    /// Apply the selection to the given record.
    bool select( const IRecord& record ) const;

    /// Clone the record selection (and any objects referenced therein).
    IRecordSelection* clone() const;

    /// Nullness operator for this selection.
    /// Returns IS_NOT_NULL for comparisons to non-NULL reference values.
    Nullness nullness() const;

    /// Relation operator for this selection.
    /// Returns EQ or NE for comparisons to NULL reference values.
    Relation relation() const;

    /// Reference value for this selection.
    /// Returns a NULL field for comparisons to NULL reference values.
    const IField& referenceValue() const;

  private:

    /// Standard constuctor is private
    FieldSelection();

    /// Copy constructor - reimplemented for the clone() method
    FieldSelection( const FieldSelection& rhs );

    /// Assignment operator is private
    FieldSelection& operator=( const FieldSelection& rhs );

    /// Initialize - extra checks for constructor of comparison to non-NULL.
    void initialize();

  private:

    /// The reference value (as a Record with one and only one field).
    /// The specification of the field is the one this selection can select.
    /// For comparisons to NULL, this field is set to NULL.
    Record m_refValue;

    /// The relation to the ref value (eg. 'GT': is this field > ref value?).
    /// For comparisons to NULL, this is set to 'EQ' or 'NE'.
    Relation m_relation;

  };

  //---------------------------------------------------------------------------

  template<typename T>
  inline FieldSelection::FieldSelection( const std::string& name,
                                         const StorageType::TypeId typeId,
                                         Relation relation,
                                         const T& refValue )
    : m_refValue( FieldSpecification( name, typeId ) )
    , m_relation( relation )
  {
    // Set the reference value and ensure that T and typeId are compatible
    m_refValue[0].setValue( refValue );
    // Extra cross-checks for constructor of comparison to non-NULL
    initialize();
  }

  //---------------------------------------------------------------------------

}
#endif // COOLKERNEL_FIELDSELECTION_H
