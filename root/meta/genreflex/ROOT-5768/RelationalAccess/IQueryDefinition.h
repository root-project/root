#ifndef RELATIONALACCESS_IQUERYDEFINITION_H
#define RELATIONALACCESS_IQUERYDEFINITION_H

#include <string>

namespace coral {

  class AttributeList;

  /**
   * Class IQueryDefinition
   * Interface for the definition of a query.
   */

  class IQueryDefinition {
  public:
    /// Set operation enumeration
    typedef enum { Union, Minus, Intersect } SetOperation;

  public:
    /**
     * Requires a distinct selection.
     */
    virtual void setDistinct() = 0;

    /**
     * Appends an expression to the output list
     */
    virtual void addToOutputList( const std::string& expression,
                                  std::string alias = "" ) = 0;

    /**
     * Appends a table name in the table selection list (the FROM part of the query)
     */
    virtual void addToTableList( const std::string& tableName,
                                 std::string alias = "" ) = 0;

    /**
     * Defines a subquery.
     * The specified name should be used in a subsequent call to addToTableList.
     */
    virtual IQueryDefinition& defineSubQuery( const std::string& allias ) = 0;

    /**
     * Defines the condition to the query (the WHERE clause)
     */
    virtual void setCondition( const std::string& condition,
                               const AttributeList& inputData ) = 0;

    /**
     * Appends a GROUP BY clause in the query
     */
    virtual void groupBy( const std::string& expression ) = 0;

    /**
     * Appends an expression to the ordering list (the ORDER clause)
     */
    virtual void addToOrderList( const std::string& expression ) = 0;

    /**
     * Instructs the server to send only up to maxRows rows at the result
     * of the query starting from the offset row.
     */
    virtual void limitReturnedRows( int maxRows = 1,
                                    int offset = 0 ) = 0;

    /**
     * Applies a set operation. Returns the rhs query definition so that it can be filled in.
     */
    virtual IQueryDefinition& applySetOperation( SetOperation opetationType ) = 0;

  protected:
    /// Protected empty destructor
    virtual ~IQueryDefinition() {}
  };

}

#endif
