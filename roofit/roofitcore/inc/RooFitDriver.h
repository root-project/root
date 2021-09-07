#ifndef ROO_FIT_DRIVER_H
#define ROO_FIT_DRIVER_H

#include "rbc.h"
#include "RooNLLVarNew.h"

#include <chrono>
#include <queue>
#include <unordered_map>

class RooAbsData;
class RooAbsArg;
class RooAbsReal;

class RooFitDriver {
  public:
     RooFitDriver(const RooAbsData& data, const RooNLLVarNew& topNode, int batchMode);
     ~RooFitDriver();
     double getVal();
     std::string const& name() const { return _name; }
     std::string const& title() const { return _title; }
     RooArgSet const& parameters() const { return _parameters; }
     double errorLevel() const { return _topNode.defaultErrorLevel(); }
     
  private:
    struct NodeInfo {
      cudaEvent_t* event = nullptr;
      cudaEvent_t* eventStart = nullptr;
      cudaStream_t* stream = nullptr; 
      std::chrono::microseconds cpuTime{0};
      std::chrono::microseconds cudaTime{std::chrono::microseconds::max()};
      std::chrono::microseconds timeLaunched{-1};
      int nClients = 0;
      int nServers = 0;
      int remClients = 0;
      int remServers = 0;
      bool computeInScalarMode = false;
      bool computeInGPU = false;
      bool copyAfterEvaluation = false;
      ~NodeInfo() {
        if (event)      rbc::dispatchCUDA->deleteCudaEvent(event);
        if (eventStart) rbc::dispatchCUDA->deleteCudaEvent(eventStart);
        if (stream)     rbc::dispatchCUDA->deleteCudaStream(stream);
      }
    };
    void updateMyClients(const RooAbsReal* node);
    void updateMyServers(const RooAbsReal* node);
    void handleIntegral(const RooAbsReal* node);
    std::pair<std::chrono::microseconds, std::chrono::microseconds> memcpyBenchmark();
    void markGPUNodes();
    void assignToGPU(const RooAbsReal* node);
    double* getAvailableCPUBuffer();
    double* getAvailableGPUBuffer();
    double* getAvailablePinnedBuffer();

    std::string _name;
    std::string _title;
    RooArgSet _parameters;

    const int _batchMode = 0;
    int _getValInvocations = 0;
    double* _cudaMemDataset = nullptr;

    // used for preserving static info about the computation graph
    rbc::DataMap _dataMapCPU;
    rbc::DataMap _dataMapCUDA;
    const RooNLLVarNew& _topNode;
    const RooAbsData* const _data = nullptr;
    const size_t _nEvents = 0;
    std::unordered_map<const RooAbsReal*, NodeInfo> _nodeInfos;

    //used for preserving resources
    std::queue<double*> _cpuBuffers;
    std::queue<double*> _gpuBuffers;
    std::queue<double*> _pinnedBuffers;
    std::vector<double> _nonDerivedValues;
};

#endif //ROO_FIT_DRIVER_H
