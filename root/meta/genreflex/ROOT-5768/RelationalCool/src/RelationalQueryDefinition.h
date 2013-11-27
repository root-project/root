// $Id: RelationalQueryDefinition.h,v 1.14 2010-03-26 18:04:14 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALQUERYDEFINITION_H
#define RELATIONALCOOL_RELATIONALQUERYDEFINITION_H

// Include files
#include <boost/shared_ptr.hpp>
#include <map>
#include "CoolKernel/Record.h"

// Local include files
#include "IRelationalQueryDefinition.h"

namespace cool 
{

  // Forward declaration
  class RelationalQueryMgr;

  /** @class RelationalQueryDefinition RelationalQueryDefinition.h
   *
   *  Relational query definition.
   *
   *  @author Andrea Valassi
   *  @date   2007-06-19
   */

  class RelationalQueryDefinition : public IRelationalQueryDefinition 
  {

  public:

    /// Constructor.
    ///
    /// if optimizeClobs is set to true, CLOBS will be first fetched as
    /// char. Only if the CLOBS are longer than 4000 characters there
    /// has to be a slower access to the actual CLOB.
    /// If optimizeClobs is set to true, you have to make sure, that
    /// checkClobLength() is called after every cursor->next()
    RelationalQueryDefinition( bool optimizeClobs = false );

    /// Destructor.
    virtual ~RelationalQueryDefinition();

    /// Clone this query definition.
    IRelationalQueryDefinition* clone() const;

    /// Get the hint in the format "/*+ hints */"
    /// ("SELECT /*+ hints */ expr1 alias1, expr2 alias2, ...").
    const std::string& getHint() const
    {
      return m_hint;
    }

    /// Get the SELECT list size
    /// ("SELECT expr1 alias1, expr2 alias2, ...").
    unsigned getSelectSize() const
    {
      return m_selectList.size();
    }

    /// Get a SELECT list item
    /// ("SELECT expr1 alias1, expr2 alias2, ...").
    const ISelectItem& getSelectItem( unsigned item ) const
    {
      return *(m_selectList[item]);
    }

    /// Get the FROM clause size
    /// ("FROM table1 alias1, table2 alias2, ...").
    unsigned getFromSize() const
    {
      return m_fromClause.size();
    }

    /// Get a FROM clause item
    /// ("FROM table1 alias1, table2 alias2, ...").
    const IFromItem& getFromItem( unsigned item ) const
    {
      return *(m_fromClause[item]);
    }

    /// Get the WHERE clause ("WHERE ...").
    const std::string& getWhereClause() const
    {
      return m_whereClause;
    }

    /// Get the GROUP BY clause size
    /// ("GROUP BY expr1, expr2...").
    unsigned getGroupSize() const
    {
      return m_groupClause.size();
    }

    /// Get a GROUP BY clause item
    /// ("GROUP BY expr1, expr2...").
    const IGroupItem& getGroupItem( unsigned item ) const
    {
      return *(m_groupClause[item]);
    }

    /// Get the ORDER BY clause size
    /// ("ORDER BY expr1, expr2...").
    unsigned getOrderSize() const
    {
      return m_orderClause.size();
    }

    /// Get an ORDER BY clause item
    /// ("ORDER BY expr1, expr2...").
    const IOrderItem& getOrderItem( unsigned item ) const
    {
      return *(m_orderClause[item]);
    }

    /// Get the bind variables.
    /// Bind variables are not necessary here if this is a subquery definition.
    const IRecord& getBindVariables() const
    {
      return m_bindVariables;
    }

    /// Get the result set specification.
    /// The result set spec is ignored if this is a subquery definition.
    const IRecordSpecification& getResultSetSpecification() const
    {
      return m_resultSetSpecification;
    }

    /// Print the relational query definition to an output stream.
    std::ostream& print( std::ostream& s ) const;

    /// Set the hint for the query: "SELECT /*+ hints */".
    /// The input argument should already be in the form "/*+ hints */".
    void setHint( const std::string& hint )
    {
      m_hint = hint;
    }

    /// Appends an expression to the select list: "SELECT expression alias".
    void addSelectItem( const std::string& expression,
                        const StorageType& type,
                        const std::string& alias = ""  )
    {
      addSelectItem( expression, type.id(), alias );
    }

    /// Appends an expression to the select list: "SELECT expression alias".
    void addSelectItem( const std::string& expression,
                        StorageType::TypeId typeId,
                        const std::string& alias = ""  )
    {
      addSelectItem( "", expression, typeId, alias );
    };

    /// If fromAlias is set, COOL can optimize access to CLOBs by converting
    /// them to a string. If the clob is larger than 4000 characters,
    /// COOL will automatically get the rest.
    void addSelectItem( const std::string& fromAlias,
                        const std::string& expression,
                        StorageType::TypeId typeId,
                        const std::string& alias = ""  );

    /// Appends a subquery to the select list: "SELECT (subquery) alias".
    /// This may contain bind variables whose values will be defined later.
    /// A deep copy of the subquery definition is performed.
    //void addSelectItem( const IRelationalQueryDefinition& subquery,
    //                    const std::string& alias = "" );

    /// Appends many fields to the select list: "SELECT [prefix]x1...".
    void addSelectItems( const IRecordSpecification& items,
                         const std::string& prefix = "" );

    /// Appends an item to the SELECT list.
    /// This may contain bind variables whose values will be defined later.
    /// A deep copy of the SELECT list item is performed.
    void addSelectItem( const ISelectItem& item )
    {
      m_selectList.push_back( item.clone() );
      //addSelectItem( item.expression(), item.typeId(), item.alias() );
    }

    /// Appends a table to the FROM clause: "FROM table alias".
    void addFromItem( const std::string& expression,
                      const std::string& alias = "" );

    /// Appends a subquery to the FROM clause: "FROM (subquery) alias".
    /// This may contain bind variables whose values will be defined later.
    /// A deep copy of the subquery definition is performed.
    void addFromItem( const IRelationalQueryDefinition& subquery,
                      const std::string& alias = "" );

    /// Appends an item to the FROM clause.
    /// This may contain bind variables whose values will be defined later.
    /// A deep copy of the FROM clause item is performed.
    void addFromItem( const IFromItem& item )
    {
      m_fromClause.push_back( item.clone() );
    }

    /// Defines the WHERE clause for the query.
    /// This may contain bind variables whose values will be defined later.
    void setWhereClause( const std::string& condition )
    {
      m_whereClause = condition;
    }

    /// Appends an item to the GROUP BY clause.
    void addGroupItem( const std::string& expression );

    /// Appends an item to the GROUP BY clause.
    /// A deep copy of the GROUP BY clause item is performed.
    void addGroupItem( const IGroupItem& item )
    {
      m_groupClause.push_back( item.clone() );
    }

    /// Appends an item to the ORDER BY clause.
    void addOrderItem( const std::string& expression );

    /// Appends an item to the ORDER BY clause.
    /// A deep copy of the ORDER BY clause item is performed.
    void addOrderItem( const IOrderItem& item )
    {
      m_orderClause.push_back( item.clone() );
    }

    /// Defines the bind variables.
    void setBindVariables( const IRecord& bindVariables )
    {
      m_bindVariables = bindVariables;
    }

    /// checkLengthClobs checks the length of optimized clobs
    /// should be called for clob optimized cursors after each cursor->next()
    ///
    /// For optimization purposes clobs are first fetched as char, this
    /// is only possible to a length up to 4000 characters. Since we
    /// don't know the length in advance we have to check the length
    /// and retrive the full clob in case it is longer than 4000 characters.
    /// All this is done by this method.
    void checkLengthClobs( coral::AttributeList* currentRow,
                           RelationalQueryMgr* queryMgr );

  private:

    RelationalQueryDefinition( const RelationalQueryDefinition& );
    RelationalQueryDefinition& operator=( const RelationalQueryDefinition& );

  private:

    class SelectItem;
    class FromItem;
    class GroupItem;
    class OrderItem;
    class ClobStringOpt;
    typedef boost::shared_ptr<ClobStringOpt> ClobStringOptPtr;

  private:

    bool m_optimizeClobs; // optimize CLOB access if < 4000 bytes? (bug #51429)
    /// count the optimized CLOB columns for naming the length columns
    int m_optimizedClobsCount; 

    std::string m_hint;
    std::vector<const ISelectItem* > m_selectList;
    std::vector<const IFromItem* > m_fromClause;
    std::string m_whereClause;
    std::vector<const IGroupItem*> m_groupClause;
    std::vector<const IOrderItem*> m_orderClause;
    Record m_bindVariables;
    RecordSpecification m_resultSetSpecification;

    typedef std::map< std::string, 
                      boost::shared_ptr<std::vector<ClobStringOptPtr> > > clobOptMap;
    clobOptMap m_clobStringOptList;
  };

}
#endif // RELATIONALCOOL_RELATIONALQUERYDEFINITION_H
