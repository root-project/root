#ifndef ROO_FIT_DRIVER_H
#define ROO_FIT_DRIVER_H

#include "rbc.h"
#include "RooNLLVarNew.h"

#include <queue>
#include <unordered_map>

class RooAbsData;
class RooAbsArg;
class RooAbsReal;

class RooFitDriver {
  public:
     RooFitDriver(const RooAbsData& data, const RooNLLVarNew& topNode, int batchMode);
     ~RooFitDriver();
     //~  inline RooAbsReal* getTopNode() { return initialQueue.back(); }
     double getVal();
     std::string const& name() const { return _name; }
     std::string const& title() const { return _title; }
     RooArgSet const& parameters() const { return _parameters; }
     double errorLevel() const { return topNode.defaultErrorLevel(); }
     
    struct NodeInfo {
        int nServers = 0;
        int nClients = 0;
        bool dependsOnObservables = true;
        bool computeInScalarMode = false;
    };

  private:
    double* getAvailableBuffer();

    std::string _name;
    std::string _title;
    RooArgSet _parameters;

    const int batchMode=0;
    double* cudaMemDataset=nullptr;
    // used for preserving static info about the computation graph
    rbc::DataMap dataMap;
    const RooNLLVarNew& topNode;
    size_t nEvents;

    RooAbsData const* _data = nullptr;

    std::queue<const RooAbsReal*> initialQueue;

    std::unordered_map<const RooAbsArg*,NodeInfo> _nodeInfos;
    // used for dynamically scheduling each step's computations
    std::queue<const RooAbsReal*> computeQueue;

    std::queue<double*> _vectorBuffers;
};

#endif //ROO_FIT_DRIVER_H
