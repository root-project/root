#ifndef RELATIONALACCESS_ITABLEDESCRIPTION_H
#define RELATIONALACCESS_ITABLEDESCRIPTION_H

#include <string>

namespace coral {

  class IColumn;
  class IPrimaryKey;
  class IIndex;
  class IForeignKey;
  class IUniqueConstraint;

  /**
   * Class ITableDescription
   * Abstract interface for the description of a relational table.
   */
  class ITableDescription {
  public:
    /**
     * Returns the name of the table.
     */
    virtual std::string name() const = 0;

    /**
     * Returns the table type (RDBMS SPECIFIC)
     */
    virtual std::string type() const = 0;

    /**
     * Returns the name of the table space for this table.
     */
    virtual std::string tableSpaceName() const = 0;

    /**
     * Returns the number of columns in the table.
     */
    virtual int numberOfColumns() const = 0;

    /**
     * Returns the description of the column corresponding to the specified index.
     * If the index is out of range an InvalidColumnIndexException is thrown.
     */
    virtual const IColumn& columnDescription( int columnIndex ) const = 0;

    /**
     * Returns the description of the column corresponding to the specified name.
     * If the specified column name is invalid an InvalidColumnNameException is thrown.
     */
    virtual const IColumn& columnDescription( const std::string& columnName ) const = 0;

    /**
     * Returns the existence of a primary key in the table.
     */
    virtual bool hasPrimaryKey() const = 0;

    /**
     * Returns the primary key for the table. If there is no primary key a NoPrimaryKeyException is thrown.
     */
    virtual const IPrimaryKey& primaryKey() const = 0;

    /**
     * Returns the number of indices defined in the table.
     */
    virtual int numberOfIndices() const = 0;

    /**
     * Returns the index corresponding to the specified identitier.
     * If the identifier is out of range an InvalidIndexIdentifierException is thrown.
     */
    virtual const IIndex& index( int indexId ) const = 0;


    /**
     * Returns the number of foreign key constraints defined in the table.
     */
    virtual int numberOfForeignKeys() const = 0;

    /**
     * Returns the foreign key corresponding to the specified identifier.
     * In case the identifier is out of range an InvalidForeignKeyIdentifierException is thrown.
     */
    virtual const IForeignKey& foreignKey( int foreignKeyIdentifier ) const = 0;

    /**
     * Returns the number of unique constraints defined in the table.
     */
    virtual int numberOfUniqueConstraints() const = 0;

    /**
     * Returns the unique constraint for the specified identifier.
     * If the identifier is out of range an InvalidUniqueConstraintIdentifierException is thrown.
     */
    virtual const IUniqueConstraint& uniqueConstraint( int uniqueConstraintIdentifier ) const = 0;

  protected:
    /// Protected empty destructor
    virtual ~ITableDescription() {}
  };

}

#endif
