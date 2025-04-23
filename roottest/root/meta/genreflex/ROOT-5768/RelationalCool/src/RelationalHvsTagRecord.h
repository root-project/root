// $Id: RelationalHvsTagRecord.h,v 1.15 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALHVSTAGRECORD_H
#define RELATIONALCOOL_RELATIONALHVSTAGRECORD_H

// Include files
//#include "CoolKernel/HvsTagRecord.h"
#include "HvsTagRecord.h"

namespace cool {

  /** @class RelationalHvsTagRecord RelationalHvsTagRecord.h
   *
   *  Relational read-only implementation of one tag in an HVS tag tree
   *  (i.e. transient representation of one row in the HVS tag table).
   *
   *  @author Andrea Valassi
   *  @date   2006-03-07
   */

  class RelationalHvsTagRecord : public HvsTagRecord {

  public:

    /// Reinterpret a relational row retrieved from persistent storage
    static const HvsTagRecord fromRow( const coral::AttributeList& tableRow );

    /// Destructor
    virtual ~RelationalHvsTagRecord() {}

    /// AV - This throws an unknown exception within fromRow on Windows
    /// [as soon as the tableRow AL argument is accessed within fromRow]
    /// Constructor from a relational row retrieved from persistent storage
    /*
    RelationalHvsTagRecord( const coral::AttributeList& tableRow )
      : HvsTagRecord( fromRow( tableRow ) ) {}
    */

    /// Copy constructor
    /// AV - Added IHvsTagRecord to avoid gcc344 warning on copy constructor
    /// AV - To be tested: would this solve the unknown Windows exception???
    RelationalHvsTagRecord( const RelationalHvsTagRecord& rhs )
      : HvsTagRecord( rhs ) {}

    /// AV - The following is a workaround (WHY???)
    /// Constructor from an HvsTagRecord
    RelationalHvsTagRecord( const HvsTagRecord& rhs )
      : HvsTagRecord( rhs ) {}

    /*
    /// Return additional 'attributes' of the HVS tag
    /// (implementation-specific attributes not exposed in the API)
    const coral::AttributeList& tagAttributes() const;
    */

    /*
    /// Change the tag description (transient - no persistent change!)
    void setDescription( const std::string& description );
    */

  private:

    /// Standard constructor is private
    RelationalHvsTagRecord();

    /// Assignment operator is private
    RelationalHvsTagRecord& operator=( const RelationalHvsTagRecord& rhs );

  private:

    /*
    /// Additional implementation-specific 'attributes' of the HVS tag
    coral::AttributeList m_tagAttributes;
    */

  };

}

#endif // RELATIONALCOOL_RELATIONALHVSTAGRECORD_H
