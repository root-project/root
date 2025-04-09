#ifndef ALGCLUSTERSRLIST_H
#define ALGCLUSTERSRLIST_H

#include <map>
#include "TObject.h"
#include "CandStripHandle.h"

class AltCandStpProbHandle {};
class AlgClusterSRList : public TObject
{

public:
  AlgClusterSRList();
  virtual ~AlgClusterSRList();
  std::map<const CandStripHandle*, Int_t> fNNeighbors;
  std::map<CandStripHandle *, double> fLikelihoods;

  std::map<CandStripHandle *,  AltCandStpProbHandle *> fLikelihoods2;


  ClassDef(AlgClusterSRList,1)                // ClusterSRList Algorithm Class
};

#endif                                                 // ALGCLUSTERSRLIST_H
