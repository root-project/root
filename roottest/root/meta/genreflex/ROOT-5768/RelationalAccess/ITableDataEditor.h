#ifndef RELATIONALACCESS_ITABLEDATAEDITOR_H
#define RELATIONALACCESS_ITABLEDATAEDITOR_H

#include <string>

namespace coral {

  // forward declarations
  class AttributeList;
  class IOperationWithQuery;
  class IBulkOperation;
  class IBulkOperationWithQuery;

  /**
   * Class ITableDataEditor
   * Interface for the DML operations on a table.
   */
  class ITableDataEditor {
  public:
    /**
     * Constructs a buffer corresponding to a full a table row.
     */
    virtual void rowBuffer( AttributeList& buffer ) = 0;

    /**
     * Inserts a new row in the table.
     */
    virtual void insertRow( const AttributeList& dataBuffer ) = 0;

    /**
     * Returns a new IOperationWithQuery object for performing an INSERT/SELECT operation
     */
    virtual IOperationWithQuery* insertWithQuery() = 0;

    /**
     * Returns a new IBulkOperation object for performing a bulk insert operation
     * specifying the input data buffer and the number of rows that should be cached on the client.
     */
    virtual IBulkOperation* bulkInsert( const AttributeList& dataBuffer,
                                        int rowCacheSize ) = 0;

    /**
     * Returns a new IBulkOperationWithQuery object for performing an INSERT/SELECT operation
     * specifying the number of iterations that should be cached on the client.
     */
    virtual IBulkOperationWithQuery* bulkInsertWithQuery( int dataCacheSize ) = 0;

    /**
     * Updates rows in the table. Returns the number of affected rows.
     */
    virtual long updateRows( const std::string& setClause,
                             const std::string& condition,
                             const AttributeList& inputData ) = 0;

    /**
     * Returns a new IBulkOperation object for performing a bulk update operation
     */
    virtual IBulkOperation* bulkUpdateRows( const std::string& setClause,
                                            const std::string& condition,
                                            const AttributeList& inputData,
                                            int dataCacheSize ) = 0;

    /**
     * Deletes the rows in the table fulfilling the specified condition. It returns the number of rows deleted.
     */
    virtual long deleteRows( const std::string& condition,
                             const AttributeList& conditionData ) = 0;

    /**
     * Returns a new IBulkOperation for peforming a bulk delete operation.
     */
    virtual IBulkOperation* bulkDeleteRows( const std::string& condition,
                                            const AttributeList& conditionData,
                                            int dataCacheSize ) = 0;

  protected:
    /// Protected empty destructor
    virtual ~ITableDataEditor() {}
  };

}

#endif
