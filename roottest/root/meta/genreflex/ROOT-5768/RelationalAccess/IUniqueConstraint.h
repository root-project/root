#ifndef RELATIONALACCESS_IUNIQUECONSTRAINT_H
#define RELATIONALACCESS_IUNIQUECONSTRAINT_H

#include <string>
#include <vector>

namespace coral {

  /**
   * Class IUniqueConstraint
   * Interface describing a unique constaint on a table.
   */
  class IUniqueConstraint {
  public:
    /**
     * Returns the system name of the constraint
     */
    virtual std::string name() const = 0;

    /**
     * Returns the names of the columns which are used to define the unique constraint on the table.
     */
    virtual const std::vector<std::string>& columnNames() const = 0;

    /**
     * Returns the name of the tablespace where the corresponding index is created.
     */
    virtual std::string tableSpaceName() const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IUniqueConstraint() {}
  };

}

#endif
