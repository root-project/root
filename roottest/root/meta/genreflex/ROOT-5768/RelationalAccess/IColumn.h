#ifndef RELATIONALACCESS_ICOLUMN_H
#define RELATIONALACCESS_ICOLUMN_H

#include <string>

namespace coral {

  /**
   * Class IColumn
   * Interface for the description of a column in a table
   */
  class IColumn {
  public:
    /**
     * Returns the name of the column.
     */
    virtual std::string name() const = 0;

    /**
     * Returns the C++ type of the column.
     */
    virtual std::string type() const = 0;

    /**
     * Returns the column id in the table.
     */
    virtual int indexInTable() const = 0;

    /**
     * Returns the NOT-NULL-ness of the column.
     */
    virtual bool isNotNull() const = 0;

    /**
     * Returns the uniqueness of the column.
     */
    virtual bool isUnique() const = 0;

    /**
     * Returns the maximum size in bytes of the data object which can be held in this column.
     */
    virtual long size() const = 0;

    /**
     * Informs whether the size of the object is fixed or it can be variable.
     * This makes sense mostly for string types.
     */
    virtual bool isSizeFixed() const = 0;

    /**
     * Returns the name of table space for the data. This makes sence mainly for LOBs.
     */
    virtual std::string tableSpaceName() const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IColumn() {}
  };

}

#endif
