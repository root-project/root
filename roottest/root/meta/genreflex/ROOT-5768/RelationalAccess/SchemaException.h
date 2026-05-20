#ifndef RELATIONALACCESS_SCHEMA_EXCEPTION_H
#define RELATIONALACCESS_SCHEMA_EXCEPTION_H

#include "SessionException.h"

namespace coral {

  /// Base class for exceptions related to accessing and modifying the schema or its data
  class SchemaException : public SessionException
  {
  public:
    /// Constructor
    SchemaException( const std::string& moduleName,
                     const std::string& message,
                     const std::string& methodName ) :
      SessionException( message, methodName, moduleName )
    {}

    SchemaException() {}

    /// Destructor
    ~SchemaException() throw() override {}
  };


  /// Exception thrown when an existing table is attempted to be recreated
  class TableAlreadyExistingException : public SchemaException
  {
  public:
    /// Constructor
    TableAlreadyExistingException( const std::string& moduleName,
                                   const std::string& tableName ) :
      SchemaException( moduleName,
                       "A table named \"" + tableName + "\" already exists",
                       "ISchema::createTable" )
    {}

    TableAlreadyExistingException() {}

    /// Destructor
    ~TableAlreadyExistingException() throw() override {}
  };


  /// Exception thrown when a non-existing table is attempted to be accessed or dropped
  class TableNotExistingException : public SchemaException
  {
  public:
    /// Constructor
    TableNotExistingException( const std::string& moduleName,
                               const std::string& tableName,
                               std::string method = "tableHandle" ) :
      SchemaException( moduleName,
                       "A table named \"" + tableName + "\" does not exist",
                       "ISchema::" + method )
    {}

    TableNotExistingException() {}

    /// Destructor
    ~TableNotExistingException() throw() override {}
  };


  /// Exception thrown when an existing view is attempted to be recreated
  class ViewAlreadyExistingException : public SchemaException
  {
  public:
    /// Constructor
    ViewAlreadyExistingException( const std::string& moduleName,
                                  const std::string& viewName ) :
      SchemaException( moduleName,
                       "A view named \"" + viewName + "\" already exists",
                       "IViewFactory::create" )
    {}

    ViewAlreadyExistingException() {}

    /// Destructor
    ~ViewAlreadyExistingException() throw() override {}
  };


  /// Exception thrown when a non-existing view is attempted to be accessed or dropped
  class ViewNotExistingException : public SchemaException
  {
  public:
    /// Constructor
    ViewNotExistingException( const std::string& moduleName,
                              const std::string& viewName,
                              std::string method = "viewHandle" ) :
      SchemaException( moduleName,
                       "A view named \"" + viewName + "\" does not exist",
                       "ISchema::" + method )
    {}

    ViewNotExistingException() {}

    /// Destructor
    ~ViewNotExistingException() throw() override {}
  };


  /// Base exception class thrown for query-related errors
  class QueryException : public SchemaException
  {
  public:
    /// Constructor
    QueryException( const std::string& moduleName,
                    const std::string& message,
                    const std::string& method ) :
      SchemaException( moduleName,
                       message,
                       method )
    {}

    QueryException() {}

    /// Destructor
    ~QueryException() throw() override {}
  };


  /// Exception thrown when a method is called after the execution or definition of a query.
  class QueryExecutedException : public QueryException
  {
  public:
    /// Constructor
    QueryExecutedException( const std::string& moduleName,
                            const std::string& method ) :
      QueryException( moduleName,
                      "A query has been already executed or defined",
                      method )
    {}

    QueryExecutedException() {}

    /// Destructor
    ~QueryExecutedException() throw() override {}
  };


  /// Exception thrown whenever an invalid unique constraint identifier is supplied
  class InvalidUniqueConstraintIdentifierException : public SchemaException
  {
  public:
    /// Constructor
    InvalidUniqueConstraintIdentifierException( const std::string& moduleName,
                                                std::string method = "ITableDescription::uniqueConstraint" ) :
      SchemaException( moduleName,
                       "Unique constraint identifier invalid or out of range",
                       method )
    {}

    InvalidUniqueConstraintIdentifierException() {}

    /// Destructor
    ~InvalidUniqueConstraintIdentifierException() throw() override {}
  };


  /// Exception thrown whenever an invalid foreign key identifier is supplied
  class InvalidForeignKeyIdentifierException : public SchemaException
  {
  public:
    /// Constructor
    InvalidForeignKeyIdentifierException( const std::string& moduleName,
                                          std::string method = "ITableDescription::foreignKey" ) :
      SchemaException( moduleName,
                       "Invalid foreign key identifier",
                       method )
    {}

    InvalidForeignKeyIdentifierException() {}

    /// Destructor
    ~InvalidForeignKeyIdentifierException() throw() override {}
  };


  /// Exception thrown whenever an invalid index identifier is supplied
  class InvalidIndexIdentifierException : public SchemaException
  {
  public:
    /// Constructor
    InvalidIndexIdentifierException( const std::string& moduleName,
                                     std::string method = "ITableDescription::index" ) :
      SchemaException( moduleName,
                       "Invalid index identifier",
                       method )
    {}

    InvalidIndexIdentifierException() {}

    /// Destructor
    ~InvalidIndexIdentifierException() throw() override {}
  };


  /// Exception thrown whenever an invalid column identifier is supplied
  class InvalidColumnIndexException : public SchemaException
  {
  public:
    /// Constructor
    InvalidColumnIndexException( const std::string& moduleName,
                                 std::string method = "ITableDescription::columnDescription" ) :
      SchemaException( moduleName,
                       "Column index out of range",
                       method )
    {}

    InvalidColumnIndexException() {}

    /// Destructor
    ~InvalidColumnIndexException() throw() override {}
  };


  /// Exception thrown whenever an invalid column name is supplied
  class InvalidColumnNameException : public SchemaException
  {
  public:
    /// Constructor
    InvalidColumnNameException( const std::string& moduleName,
                                std::string method = "ITableDescription::columnDescription" ) :
      SchemaException( moduleName,
                       "Invalid or non-existing column name",
                       method )
    {}

    InvalidColumnNameException() {}

    /// Destructor
    ~InvalidColumnNameException() throw() override {}
  };


  /// Exception thrown whenever an non existing primary key is attempted to be accessed or dropped
  class NoPrimaryKeyException : public SchemaException
  {
  public:
    /// Constructor
    NoPrimaryKeyException( const std::string& moduleName,
                           std::string method = "ITableDescription::primaryKey" ) :
      SchemaException( moduleName,
                       "No primary key exists on the table",
                       method )
    {}

    NoPrimaryKeyException() {}

    /// Destructor
    ~NoPrimaryKeyException() throw() override {}
  };


  /// Exception thrown whenever an existing primary key is attempted to be recreated.
  class ExistingPrimaryKeyException : public SchemaException
  {
  public:
    /// Constructor
    ExistingPrimaryKeyException( const std::string& moduleName  ) :
      SchemaException( moduleName,
                       "A primary key already exists on the table",
                       "ITableSchemaEditor::setPrimaryKey" )
    {}

    ExistingPrimaryKeyException() {}

    /// Destructor
    ~ExistingPrimaryKeyException() throw() override {}
  };


  /// Exception thrown whenever an existing unique constraint is attempted to be overriden
  class UniqueConstraintAlreadyExistingException : public SchemaException
  {
  public:
    /// Constructor
    UniqueConstraintAlreadyExistingException( const std::string& moduleName  ) :
      SchemaException( moduleName,
                       "A unique constraint already exists for the specified set of columns",
                       "ITableSchemaEditor::setUniqueConstraint" )
    {}

    UniqueConstraintAlreadyExistingException() {}

    /// Destructor
    ~UniqueConstraintAlreadyExistingException() throw() override {}
  };


  /// Exception thrown at DML operations
  class DataEditorException : public SchemaException
  {
  public:
    /// Constructor
    DataEditorException( const std::string& moduleName,
                         const std::string& message,
                         const std::string& method ) :
      SchemaException( moduleName,
                       message,
                       method )
    {}

    DataEditorException() {}

    /// Destructor
    ~DataEditorException() throw() override {}
  };


  /// Exception thrown whenever a duplicate key is attempted to be inserted
  class DuplicateEntryInUniqueKeyException : public DataEditorException
  {
  public:
    /// Constructor
    DuplicateEntryInUniqueKeyException( const std::string& moduleName,
                                        std::string methodName = "ITableDataEditor::insertRow" ) :
      DataEditorException( moduleName,
                           "Attempted to insert a duplicate value in a unique key",
                           methodName )
    {}

    DuplicateEntryInUniqueKeyException() {}

    /// Destructor
    ~DuplicateEntryInUniqueKeyException() throw() override {}
  };

}

#endif
