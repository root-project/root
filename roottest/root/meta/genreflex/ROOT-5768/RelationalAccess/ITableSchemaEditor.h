#ifndef RELATIONALACCESS_ITABLESCHEMAEDITOR_H
#define RELATIONALACCESS_ITABLESCHEMAEDITOR_H 1

// Include files
#include <string>
#include <vector>

namespace coral {

  /**
   * Class ITableSchemaEditor
   * Interface for altering the schema of an existing table
   */
  class ITableSchemaEditor {
  public:
    /**
     * Inserts a new column in the table.
     * If the column name already exists or is invalid, an InvalidColumnNameException is thrown.
     */
    virtual void insertColumn( const std::string& name,
                               const std::string& type,
                               int size = 0,
                               bool fixedSize = true,
                               std::string tableSpaceName = "" ) = 0;

    /**
     * Drops a column from the table.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     */
    virtual void dropColumn( const std::string& name ) = 0;

    /**
     * Renames a column in the table.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     */
    virtual void renameColumn( const std::string& originalName,
                               const std::string& newName ) = 0;

    /**
     * Changes the C++ type of a column in the table.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     */
    virtual void changeColumnType( const std::string& columnName,
                                   const std::string& typeName,
                                   int size = 0,
                                   bool fixedSize = true ) = 0;

    /**
     * Sets or removes a NOT NULL constraint on a column.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     */
    virtual void setNotNullConstraint( const std::string& columnName,
                                       bool isNotNull = true ) = 0;

    /**
     * Adds or removes a unique constraint on a column.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If a unique constrain already exists for the specified column an UniqueConstraintAlreadyExistingException is thrown.
     * If not unique constraint exists for the specified column in the case it is asked to be dropped,
     * an InvalidUniqueConstraintIdentifierException is thrown
     */
    virtual void setUniqueConstraint( const std::string& columnName,
                                      std::string name = "",
                                      bool isUnique = true,
                                      std::string tableSpaceName = "" ) = 0;

    /**
     * Adds or removes a unique constraint defined over one or more columns.
     * If any of the column names does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If a unique constrain already exists for the specified set of columns an UniqueConstraintAlreadyExistingException is thrown.
     * If not unique constraint exists for the specified set of columns in the case it is asked to be dropped,
     * an InvalidUniqueConstraintIdentifierException is thrown
     */
    virtual void setUniqueConstraint( const std::vector<std::string>& columnNames,
                                      std::string name = "",
                                      bool isUnique = true,
                                      std::string tableSpaceName = "" ) = 0;

    /**
     * Defines a primary key from a single column.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If a primary key has already been defined, an ExistingPrimaryKeyException is thrown.
     */
    virtual void setPrimaryKey( const std::string& columnName,
                                std::string tableSpaceName = "" ) = 0;

    /**
     * Defines a primary key from one or more columns.
     * If any of the column names does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If a primary key has already been defined, an ExistingPrimaryKeyException is thrown.
     */
    virtual void setPrimaryKey( const std::vector<std::string>& columnNames,
                                std::string tableSpaceName = "" ) = 0;

    /**
     * Drops the existing primary key.
     * If there is no primary key defined a NoPrimaryKeyException is thrown.
     */
    virtual void dropPrimaryKey() = 0;

    /**
     * Creates an index on a column.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If an index has already been defined with that name an InvalidIndexIdentifierException is thrown.
     */
    virtual void createIndex( const std::string& indexName,
                              const std::string& columnName,
                              bool isUnique = false,
                              std::string tableSpaceName = "" ) = 0;

    /**
     * Creates an index over one or more columns.
     * If any of the column names does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If an index has already been defined with that name an InvalidIndexIdentifierException is thrown.
     */
    virtual void createIndex( const std::string& name,
                              const std::vector<std::string>& columnNames,
                              bool isUnique = false,
                              std::string tableSpaceName = "" ) = 0;

    /**
     * Drops an existing index.
     * If the specified index name is not valid an InvalidIndexIdentifierException is thrown.
     */
    virtual void dropIndex( const std::string& indexName ) = 0;

    /**
     * Creates a foreign key constraint.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If a foreign key has already been defined with that name an InvalidForeignKeyIdentifierException is thrown.
     */
    virtual void createForeignKey( const std::string& name,
                                   const std::string& columnName,
                                   const std::string& referencedTableName,
                                   const std::string& referencedColumnName ) = 0;

    /**
     * Creates a foreign key over one or more columns.
     * If any of the column names does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If a foreign key has already been defined with that name an InvalidForeignKeyIdentifierException is thrown.
     */
    virtual void createForeignKey( const std::string& name,
                                   const std::vector<std::string>& columnNames,
                                   const std::string& referencedTableName,
                                   const std::vector<std::string>& referencedColumnNames ) = 0;

    /**
     * Drops a foreign key.
     * If the specified name is not valid an InvalidForeignKeyIdentifierException is thrown.
     */
    virtual void dropForeignKey( const std::string& name ) = 0;

  protected:
    /// Protected empty destructor
    virtual ~ITableSchemaEditor() {}
  };

}

#endif
