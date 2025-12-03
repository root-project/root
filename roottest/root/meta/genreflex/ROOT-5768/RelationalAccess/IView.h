#ifndef RELATIONALACCESS_IVIEW_H
#define RELATIONALACCESS_IVIEW_H

#include <string>

namespace coral {

  // forward declarations
  class IColumn;
  class ITablePrivilegeManager;

  /**
   * Class IView
   * Interface for a view
   */
  class IView {
  public:
    /**
     * Returns the view name
     */
    virtual std::string name() const = 0;

    /**
     * Returns the SQL string defining the view.
     * The SQL string is RDBMS-specific.
     */
    virtual std::string definition() const = 0;

    /**
     * Returns the number of columns in the view
     */
    virtual int numberOfColumns() const = 0;

    /**
     * Returns a reference to a column description object for the specified column index.
     * If the specified index is out of range, an InvalidColumnIndexException is thrown.
     */
    virtual const IColumn& column( int index ) const = 0;

    /**
     * Returns a reference to the privilege manager of the view.
     */
    virtual ITablePrivilegeManager& privilegeManager() = 0;

  protected:
    /// Protected empty destructor
    virtual ~IView() {}
  };

}

#endif
