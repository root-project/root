#ifndef ALGCLUSTERSRLIST_H
#define ALGCLUSTERSRLIST_H

#include <map>
#include "TObject.h"
#include "CandStripHandle.h"

class AlgClusterSRList : public TObject
{

public:
  AlgClusterSRList();
  virtual ~AlgClusterSRList();
  std::map<const CandStripHandle*, Int_t> fNNeighbors;

ClassDef(AlgClusterSRList,1)                // ClusterSRList Algorithm Class
};

#endif                                                 // ALGCLUSTERSRLIST_H
