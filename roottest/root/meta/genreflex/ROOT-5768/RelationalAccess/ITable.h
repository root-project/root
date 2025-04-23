#ifndef RELATIONALACCESS_ITABLE_H
#define RELATIONALACCESS_ITABLE_H

namespace coral {

  // forward declarations
  class ITableDescription;
  class ITableSchemaEditor;
  class ITableDataEditor;
  class ITablePrivilegeManager;
  class IQuery;

  /**
   * Class ITable
   * Interface for accessing and manipulating the data and the description of a relational table.
   */
  class ITable {
  public:
    /**
     * Returns the description of the table.
     */
    virtual const ITableDescription& description() const = 0;

    /**
     * Returns a reference to the schema editor for the table.
     */
    virtual ITableSchemaEditor& schemaEditor() = 0;

    /**
     * Returns a reference to the ITableDataEditor object  for the table.
     */
    virtual ITableDataEditor& dataEditor() = 0;

    /**
     * Returns a reference to the privilege manager of the table.
     */
    virtual ITablePrivilegeManager& privilegeManager() = 0;

    /**
     * Returns a new query object for performing a query involving this table only.
     */
    virtual IQuery* newQuery() const = 0;

  protected:
    /// Protected empty destructor
    virtual ~ITable() {}
  };

}

#endif
