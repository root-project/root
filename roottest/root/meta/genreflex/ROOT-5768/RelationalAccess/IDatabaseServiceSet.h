#ifndef RELATIONALACCESS_IDATABASESERVICESET_H
#define RELATIONALACCESS_IDATABASESERVICESET_H

namespace coral {

  // forward declarations
  class IDatabaseServiceDescription;

  /**
   * Class IDatabaseServiceSet
   *
   * Container for the available database services corresponding to
   * a logical database connection. The services should be tried out
   * with the order which they appear in the set.
   */
  class IDatabaseServiceSet {
  public:
    /// Destructor
    virtual ~IDatabaseServiceSet() {}

    /**
     * Returns the number of actual database services corresponding to
     * the logical name.
     */
    virtual int numberOfReplicas() const = 0;

    /**
     * Returns a reference to the service description object corresponding
     * to the specified index.
     * If the index is out of range an InvalidReplicaIdentifierException is thrown.
     */
    virtual const IDatabaseServiceDescription& replica( int index ) const = 0;
  };

}

#endif
