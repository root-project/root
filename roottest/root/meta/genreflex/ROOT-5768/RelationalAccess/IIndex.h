#ifndef RELATIONALACCESS_IINDEX_H
#define RELATIONALACCESS_IINDEX_H

#include <string>
#include <vector>

namespace coral {

  /**
   * Class IIndex
   * Interface describing an index on a table.
   */
  class IIndex {
  public:
    /**
     * Returns the system name of the index.
     */
    virtual std::string name() const = 0;

    /**
     * Returns the names of the columns which compose the index.
     */
    virtual const std::vector<std::string>& columnNames() const = 0;

    /**
     * Returns the uniqueness of the index.
     */
    virtual bool isUnique() const = 0;

    /**
     * Returns the name of the table space where the index is created.
     */
    virtual std::string tableSpaceName() const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IIndex() {}
  };

}

#endif
