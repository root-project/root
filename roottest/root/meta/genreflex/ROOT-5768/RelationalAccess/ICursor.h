#ifndef RELATIONALACCESS_ICURSOR_H
#define RELATIONALACCESS_ICURSOR_H

namespace coral {

  class AttributeList;

  /**
   * Class ICursor
   *
   * Interface for the iteration over the result set of a query
   */
  class ICursor {
  public:
    /**
     * Positions the cursor to the next available row in the result set.
     * If there are no more rows in the result set false is returned.
     */
    virtual bool next() = 0;

    /**
     * Returns a reference to output buffer holding the data of the last
     * row fetched.
     */
    virtual const AttributeList& currentRow() const = 0;

    /**
     * Explicitly closes the cursor, releasing the resources on the server.
     */
    virtual void close() = 0;

  protected:
    /// Protected empty destructor
    virtual ~ICursor() {}
  };

}
#endif
