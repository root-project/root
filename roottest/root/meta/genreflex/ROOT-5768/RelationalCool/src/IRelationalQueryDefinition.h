// $Id: IRelationalQueryDefinition.h,v 1.11 2010-03-24 16:01:33 avalassi Exp $
#ifndef RELATIONALCOOL_IRELATIONALQUERYDEFINITION_H
#define RELATIONALCOOL_IRELATIONALQUERYDEFINITION_H

// Include files
#include <iostream>
#include <string>
#include <vector>
#include "CoolKernel/IRecord.h"

namespace cool 
{

  // Forward declaration
  class RelationalQueryMgr;

  //--------------------------------------------------------------------------

  /** @class RelationalQueryDefinition RelationalQueryDefinition.h
   *
   *  Abstract interface to a relational query definition.
   *
   *  @author Andrea Valassi
   *  @date   2007-06-19
   */

  class IRelationalQueryDefinition 
  {

  public:

    class ISelectItem
    {
    public:
      virtual ~ISelectItem() {}
      virtual bool isSubquery() const = 0;
      virtual const std::string& expression() const = 0;
      //virtual const IRelationalQueryDefinition& subquery() const = 0;
      virtual const std::string& alias() const = 0;
      virtual StorageType::TypeId typeId() const = 0;
      virtual const std::string& fromAlias() const = 0;
      virtual const ISelectItem* clone() const = 0;
    };

    class IFromItem
    {
    public:
      virtual ~IFromItem() {}
      virtual bool isSubquery() const = 0;
      virtual const std::string& expression() const = 0;
      virtual const IRelationalQueryDefinition& subquery() const = 0;
      virtual const std::string& alias() const = 0;
      virtual const IFromItem* clone() const = 0;
    };

    class IGroupItem
    {
    public:
      virtual ~IGroupItem() {}
      virtual const std::string& expression() const = 0;
      virtual const IGroupItem* clone() const = 0;
    };

    class IOrderItem
    {
    public:
      virtual ~IOrderItem() {}
      virtual const std::string& expression() const = 0; // may end in ASC/DESC
      virtual const IOrderItem* clone() const = 0;
    };

    /// Destructor.
    virtual ~IRelationalQueryDefinition() {}

    /// Clone this query definition.
    virtual IRelationalQueryDefinition* clone() const = 0;

    /// Get the hint in the format "/*+ hints */"
    /// ("SELECT /*+ hints */ expr1 alias1, expr2 alias2, ...").
    virtual const std::string& getHint() const = 0;

    /// Get the SELECT list size
    /// ("SELECT expr1 alias1, expr2 alias2, ...").
    virtual unsigned getSelectSize() const = 0;

    /// Get a SELECT list item
    /// ("SELECT expr1 alias1, expr2 alias2, ...").
    virtual const ISelectItem& getSelectItem( unsigned item ) const = 0;

    /// Get the FROM clause size
    /// ("FROM table1 alias1, table2 alias2, ...").
    virtual unsigned getFromSize() const = 0;

    /// Get a FROM clause item
    /// ("FROM table1 alias1, table2 alias2, ...").
    virtual const IFromItem& getFromItem( unsigned item ) const = 0;

    /// Get the WHERE clause ("WHERE ...").
    virtual const std::string& getWhereClause() const = 0;

    /// Get the GROUP BY clause size
    /// ("GROUP BY expr1, expr2...").
    virtual unsigned getGroupSize() const = 0;

    /// Get a GROUP BY clause item
    /// ("GROUP BY expr1, expr2...").
    virtual const IGroupItem& getGroupItem( unsigned item ) const = 0;

    /// Get the ORDER BY clause size
    /// ("ORDER BY expr1, expr2...").
    virtual unsigned getOrderSize() const = 0;

    /// Get a ORDER BY clause item
    /// ("ORDER BY expr1, expr2...").
    virtual const IOrderItem& getOrderItem( unsigned item ) const = 0;

    /// Get the bind variables.
    /// Bind variables are not necessary here if this is a subquery definition.
    virtual const IRecord& getBindVariables() const = 0;

    /// Get the result set specification.
    /// The result set spec is ignored if this is a subquery definition.
    virtual const IRecordSpecification& getResultSetSpecification() const = 0;

    /// Print the relational query definition to an output stream.
    virtual std::ostream& print( std::ostream& s ) const = 0;

    /// check the length of clobs fetch as char, and retrive it fully in case
    /// it is longer than 4000 characters
    virtual void checkLengthClobs( coral::AttributeList* currentRow,
                                   RelationalQueryMgr* queryMgr ) = 0;
  };

  //--------------------------------------------------------------------------

  inline std::ostream& operator<<( std::ostream& s,
                                   const IRelationalQueryDefinition& def )
  {
    return def.print( s );
  }

  //--------------------------------------------------------------------------

}

#endif // RELATIONALCOOL_IRELATIONALQUERYDEFINITION_H
