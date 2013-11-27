#ifndef RELATIONALACCESS_IPRIMARYKEY_H
#define RELATIONALACCESS_IPRIMARYKEY_H

#include <string>
#include <vector>

namespace coral {

  /**
   * Class IPrimaryKey
   * Interface for the description of a primary key in a table.
   */
  class IPrimaryKey {
  public:
    /**
     * Returns the names of the columns which form the primary key.
     */
    virtual const std::vector<std::string>& columnNames() const = 0;

    /**
     * Returns the name of the table space where the corresponding index is created.
     */
    virtual std::string tableSpaceName() const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IPrimaryKey() {}
  };

}

#endif
