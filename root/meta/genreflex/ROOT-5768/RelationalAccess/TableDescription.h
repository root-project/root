#ifndef RELATIONALACCESS_TABLEDESCRIPTION_H
#define RELATIONALACCESS_TABLEDESCRIPTION_H 1

// Include files
#include "RelationalAccess/ITableSchemaEditor.h"
#include "RelationalAccess/ITableDescription.h"

namespace coral {

  // forward declarations
  class Column;
  class UniqueConstraint;
  class PrimaryKey;
  class Index;
  class ForeignKey;

  /**
   * Class TableDescription
   *
   * Transient implementation of the ITableDescription which can be passed as an
   * argument to the createTable method of the ISchema class.
   */
  class TableDescription : virtual public ITableSchemaEditor,
                           virtual public ITableDescription
  {
  public:
    /// Constructor
    explicit TableDescription( std::string context = "User" );

    /// Copy constructor.
    /// Deep copy everything, including columns, PK, UKs, FKs, indexes.
    TableDescription( const TableDescription& rhs );

    /// Assignment operator.
    /// Deep copy everything, including columns, PK, UKs, FKs, indexes.
    TableDescription& operator=( const TableDescription& rhs );

    /**
     * Constructor from the base class.
     * It does copies over the definition of the columns, and unique constraints
     * but not those of the foreign keys or indices because their names cannot be copied.
     */
    TableDescription( const coral::ITableDescription& rhs,
                      std::string context = "User" );

    /// Destructor
    virtual ~TableDescription();

    /**
     * Sets the name of the table
     */
    void setName( const std::string& tableName );

    /**
     * Sets the type of the table
     */
    void setType( const std::string& tableType );

    /**
     * Sets the name of the table space
     */
    void setTableSpaceName( const std::string& tableSpaceName );




    // Methods inherited of ITableSchemaEditor

    /**
     * Inserts a new column in the table.
     * If the column name already exists or is invalid, an InvalidColumnNameException is thrown.
     */
    void insertColumn( const std::string& name,
                       const std::string& type,
                       int size = 0,
                       bool fixedSize = true,
                       std::string tableSpaceName = "" );

    /**
     * Drops a column from the table.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     */
    void dropColumn( const std::string& name );

    /**
     * Renames a column in the table.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     */
    void renameColumn( const std::string& originalName,
                       const std::string& newName );

    /**
     * Changes the C++ type of a column in the table.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     */
    void changeColumnType( const std::string& columnName,
                           const std::string& typeName,
                           int size = 0,
                           bool fixedSize = true );

    /**
     * Sets or removes a NOT NULL constraint on a column.
     * If the column name already exists or is invalid, an InvalidColumnNameException is thrown.
     */
    void setNotNullConstraint( const std::string& columnName,
                               bool isNotNull = true );

    /**
     * Adds or removes a unique constraint on a column.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     *
     */
    void setUniqueConstraint( const std::string& columnName,
                              std::string name = "",
                              bool isUnique = true,
                              std::string tableSpaceName = "" );

    /**
     * Adds or removes a unique constraint defined over one or more columns.
     * If any of the column names does not exist or is invalid, an InvalidColumnNameException is thrown.
     *
     */
    void setUniqueConstraint( const std::vector<std::string>& columnNames,
                              std::string name = "",
                              bool isUnique = true,
                              std::string tableSpaceName = "" );

    /**
     * Defines a primary key from a single column.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If a primary key has already been defined, an ExistingPrimaryKeyException is thrown.
     */
    void setPrimaryKey( const std::string& columnName,
                        std::string tableSpaceName = "" );

    /**
     * Defines a primary key from one or more columns.
     * If any of the column names does not exist or is invalid, an InvalidColumnNameException is thrown.
     * If a primary key has already been defined, an ExistingPrimaryKeyException is thrown.
     */
    void setPrimaryKey( const std::vector<std::string>& columnNames,
                        std::string tableSpaceName = "" );

    /**
     * Drops the existing primary key.
     * If there is no primary key defined a NoPrimaryKeyException is thrown.
     */
    void dropPrimaryKey();

    /**
     * Creates an index on a column.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     *
     */
    void createIndex( const std::string& indexName,
                      const std::string& columnName,
                      bool isUnique = false,
                      std::string tableSpaceName = "" );

    /**
     * Creates an index over one or more columns.
     * If any of the column names does not exist or is invalid, an InvalidColumnNameException is thrown.
     *
     */
    void createIndex( const std::string& name,
                      const std::vector<std::string>& columnNames,
                      bool isUnique = false,
                      std::string tableSpaceName = "" );

    /**
     * Drops an existing index.
     * If the specified index name is not valid an InvalidIndexIdentifierException is thrown.
     */
    void dropIndex( const std::string& indexName );

    /**
     * Creates a foreign key constraint.
     * If the column name does not exist or is invalid, an InvalidColumnNameException is thrown.
     *
     */
    void createForeignKey( const std::string& name,
                           const std::string& columnName,
                           const std::string& referencedTableName,
                           const std::string& referencedColumnName );

    /**
     * Creates a foreign key over one or more columns.
     * If any of the column names does not exist or is invalid, an InvalidColumnNameException is thrown.
     *
     */
    void createForeignKey( const std::string& name,
                           const std::vector<std::string>& columnNames,
                           const std::string& referencedTableName,
                           const std::vector<std::string>& referencedColumnNames );

    /**
     * Drops a foreign key.
     * If the specified name is not valid an InvalidForeignKeyIdentifierException is thrown.
     */
    void dropForeignKey( const std::string& name );





    // Methods inherited of ITableDescription

    /**
     * Returns the name of the table.
     */
    std::string name() const;

    /**
     * Returns the table type (RDBMS SPECIFIC)
     */
    std::string type() const;

    /**
     * Returns the name of the table space for this table.
     */
    std::string tableSpaceName() const;

    /**
     * Returns the number of columns in the table.
     */
    int numberOfColumns() const;

    /**
     * Returns the description of the column corresponding to the specified index.
     * If the index is out of range an InvalidColumnIndexException is thrown.
     */
    const IColumn& columnDescription( int columnIndex ) const;

    /**
     * Returns the description of the column corresponding to the specified name.
     * If the specified column name is invalid an InvalidColumnNameException is thrown.
     */
    const IColumn& columnDescription( const std::string& columnName ) const;

    /**
     * Returns the existence of a primary key in the table.
     */
    bool hasPrimaryKey() const;

    /**
     * Returns the primary key for the table. If there is no primary key a NoPrimaryKeyException is thrown.
     */
    const IPrimaryKey& primaryKey() const;

    /**
     * Returns the number of indices defined in the table.
     */
    int numberOfIndices() const;

    /**
     * Returns the index corresponding to the specified identitier.
     * If the identifier is out of range an InvalidIndexIdentifierException is thrown.
     */
    const IIndex& index( int indexId ) const;


    /**
     * Returns the number of foreign key constraints defined in the table.
     */
    int numberOfForeignKeys() const;

    /**
     * Returns the foreign key corresponding to the specified identifier.
     * In case the identifier is out of range an InvalidForeignKeyIdentifierException is thrown.
     */
    const IForeignKey& foreignKey( int foreignKeyIdentifier ) const;

    /**
     * Returns the number of unique constraints defined in the table.
     */
    int numberOfUniqueConstraints() const;

    /**
     * Returns the unique constraint for the specified identifier.
     * If the identifier is out of range an InvalidUniqueConstraintIdentifierException is thrown.
     */
    const IUniqueConstraint& uniqueConstraint( int uniqueConstraintIdentifier ) const;

  private:

    /// Clear the table description
    void clear();

  private:

    /// The context string
    std::string m_context;

    /// The table name
    std::string m_name;

    /// The table type
    std::string m_type;

    /// The table space name
    std::string m_tableSpaceName;

    /// The columns
    std::vector< Column* > m_columns;

    /// The unique constraints
    std::vector< UniqueConstraint* > m_uniqueConstraints;

    /// The primary key
    PrimaryKey* m_primaryKey;

    /// The indices
    std::vector< Index* > m_indices;

    /// The foreign keys
    std::vector< ForeignKey* > m_foreignKeys;
  };

}

#endif
