#ifndef RELATIONALACCESS_IREPLICASORTINGALGORITHM_H
#define RELATIONALACCESS_IREPLICASORTINGALGORITHM_H

#include <vector>

namespace coral {

  class IDatabaseServiceDescription;

  /**
   * Class IReplicaSortingAlgorithm
   *
   * Interface for defining the replica sorting algorithm
   */
  class IReplicaSortingAlgorithm {

  public:
    /// Empty destructor
    virtual ~IReplicaSortingAlgorithm(){}

    // ownership of IDatabaseServiceDescription instances is left to the caller
    virtual void sort(std::vector<const IDatabaseServiceDescription*>& replicaSet) = 0;

  };

}

#endif
