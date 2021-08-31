#include "RooFitDriver.h"
#include "RooAbsData.h"
#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooBatchCompute.h"
#include "RooNLLVarNew.h"
#include "RooRealVar.h"
#include "RunContext.h"

#include <chrono>
#include <thread>

RooFitDriver::RooFitDriver(const RooAbsData& data, const RooNLLVarNew& topNode, int batchMode)
  : _name{topNode.GetName()}, _title{topNode.GetTitle()}
  , _parameters{*std::unique_ptr<RooArgSet>(topNode.getParameters(data, true))}
  , _batchMode{batchMode}, _topNode{topNode}
  , _data{&data}
  , _nEvents{static_cast<size_t>( data.numEntries() )}
{
  // fill the RunContext with the observable data and map the observables
  // by namePtr in order to replace their memory addresses later, with
  // the ones from the variables that are actually in the computation graph. 
  rbc::RunContext evalData;
  data.getBatches(evalData, 0, _nEvents);
  _dataMapCPU = evalData.spans;
  std::unordered_map<const TNamed*,const RooAbsReal*> nameResolver;
  for (auto& it:_dataMapCPU) nameResolver[it.first->namePtr()]=it.first;

  // Check if there is a batch for weights and if it's already in the dataMap.
  // If not, we need to put the batch and give as a key a RooRealVar* that has
  // the same name as RooNLLVarNew's _weight proxy, so that it gets renamed like
  // every other observable.
  RooSpan<const double> weights = data.getWeightBatch(0, _nEvents);
  std::string weightVarName = data.getWeightVarName()!="" ? data.getWeightVarName() : "_weight";
  RooRealVar dummy (weightVarName.c_str(), "dummy", 0.0);
  const TNamed* pTNamed = dummy.namePtr();
  if (!weights.empty() && nameResolver.count(pTNamed)==0)
  {
    _dataMapCPU[&dummy] = weights;
    nameResolver[pTNamed] = &dummy;
  }

  // Get a serial list of the nodes in the computation graph.
  // treeNodeServelList() is recursive and adds the top node before the children,
  // so reversing the list gives us a topological ordering of the graph.
  RooArgList list;
  _topNode.treeNodeServerList(&list);
  for (int i=list.size()-1; i>=0; i--)
  {
    auto pAbsReal = dynamic_cast<RooAbsReal*>(&list[i]);
    if (!pAbsReal) continue;
    const bool alreadyExists = nameResolver.count(pAbsReal->namePtr());
    const RooAbsReal* pClone = nameResolver[pAbsReal->namePtr()];
    if (alreadyExists && !pClone) continue; // node included multiple times in the list
      
    if (pClone) //this node is an observable, update the RunContext and don't add it in `_nodeInfos`.
    {
      auto it = _dataMapCPU.find(pClone);
      _dataMapCPU[pAbsReal]=it->second;
      _dataMapCPU.erase(it);

      // set nameResolver to nullptr to be able to detect future duplicates
      nameResolver[pAbsReal->namePtr()] = nullptr;
    }
    else //this node needs evaluation, mark it's clients
    {
      // If the node doesn't depend on any observables, there is no need to
      // loop over events and we don't need to use the batched evaluation.
      RooArgSet observablesForNode;
      pAbsReal->getObservables(_data->get(), observablesForNode);
      _nodeInfos[pAbsReal].computeInScalarMode = observablesForNode.empty() || !pAbsReal->isDerived();

      auto clients = pAbsReal->valueClients();
      for (auto* client:clients)
        if(list.find(*client))
        {
          auto pClient = static_cast<const RooAbsReal*>(client);
          ++_nodeInfos[pClient].nServers;
          ++_nodeInfos[pAbsReal].nClients;
        }
    }
  }

  // Extra steps for initializing in cuda mode
  if (_batchMode != -1) return;
  if (!rbc::dispatch_gpu) 
    throw std::runtime_error(std::string("In: ")+__func__+"(), "+__FILE__+":"+__LINE__+": Cuda implementation of the computing library is not available\n");

  markGPUNodes();

  // copy observable data to the gpu
  _cudaMemDataset = static_cast<double*>(rbc::dispatch_gpu->cudaMalloc( _nEvents*_dataMapCPU.size()*sizeof(double) ));
  size_t idx=0;
  for (auto& record:_dataMapCPU)
  {
    _dataMapCUDA[record.first] = RooSpan<double>(_cudaMemDataset+idx, _nEvents);
    rbc::dispatch_gpu->memcpyToGPU(_cudaMemDataset+idx, record.second.data(), _nEvents*sizeof(double));
    idx += _nEvents;
  }
}

namespace {

template<typename T, typename Func_t>
T getAvailable(std::queue<T>& q, Func_t creator) {
  if (q.empty()) return static_cast<T>(creator());
  else {
    T ret = q.front();
    q.pop();
    return ret;
  }
}

template<typename T, typename Func_t>
void clearQueue(std::queue<T>& q, Func_t destroyer) {
  while (!q.empty())
  {
    destroyer(q.front());
    q.pop();
  }
}
} // end anonymous namespace

RooFitDriver::~RooFitDriver()
{
  clearQueue(_cpuBuffers,      [](double* ptr){delete[] ptr;} );
  if (_batchMode==-1)
  {
    clearQueue(_gpuBuffers,    [](double* ptr){rbc::dispatch_gpu->cudaFree(ptr);} );
    clearQueue(_pinnedBuffers, [](double* ptr){rbc::dispatch_gpu->cudaFreeHost(ptr);} );
    rbc::dispatch_gpu->cudaFree(_cudaMemDataset);
  }
}

double* RooFitDriver::getAvailableCPUBuffer() {
  return getAvailable(_cpuBuffers, [=](){return new double[_nEvents];} );
}
double* RooFitDriver::getAvailableGPUBuffer() {
  return getAvailable(_gpuBuffers, [=](){return rbc::dispatch_gpu->cudaMalloc(_nEvents*sizeof(double));} );
}
double* RooFitDriver::getAvailablePinnedBuffer() {
  return getAvailable(_pinnedBuffers, [=](){return rbc::dispatch_gpu->cudaMallocHost(_nEvents*sizeof(double));} );
}

double RooFitDriver::getVal()
{
  for (auto& item:_nodeInfos) {
    item.second.remClients = item.second.nClients;
    item.second.remServers = item.second.nServers;
  }
  _nonDerivedValues.clear();
  _nonDerivedValues.reserve(_nodeInfos.size()); // to avoid reallocation
  
  // find initial gpu nodes and assign them to gpu
  for (const auto& it:_nodeInfos)
    if (it.second.remServers==0 && it.second.computeInGPU)
      assignToGPU(it.first);
  
  int nNodes = _nodeInfos.size();
  while (nNodes)
  {
    // find finished gpu nodes
    if (_batchMode==-1)
      for (auto& it:_nodeInfos)
        if (it.second.remServers==-1 && !rbc::dispatch_gpu->streamIsActive(it.second.stream))
        {
          it.second.remServers=-2;
          nNodes--;
          updateMyClients(it.first);
          updateMyServers(it.first);
        }
    
    // find next cpu node
    auto it=_nodeInfos.begin();
    for ( ; it!=_nodeInfos.end(); it++)
      if (it->second.remServers==0 && !it->second.computeInGPU) break;

    // if no cpu node available sleep for a while to save cpu usage
    if (it==_nodeInfos.end())
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    
    // compute next cpu node
    const RooAbsReal* node = it->first;
    NodeInfo& info = it->second;
    info.remServers=-2; //so that it doesn't get picked again
    nNodes--;
    if (info.computeInScalarMode)
    {
      _nonDerivedValues.push_back(node->getVal(_data->get()));
      _dataMapCPU[node] = _dataMapCUDA[node] = RooSpan<const double>(&_nonDerivedValues.back(),1);
    }
    else
    {     
      double* buffer = info.copyAfterEvaluation ? getAvailablePinnedBuffer() : getAvailableCPUBuffer();
      _dataMapCPU[node] = RooSpan<const double>(buffer, _nEvents);
      handleIntegral(node);
      rbc::dispatch = rbc::dispatch_cpu;
      node->computeBatch(buffer, _nEvents, _dataMapCPU);
      rbc::dispatch = nullptr;
      if (info.copyAfterEvaluation) 
      {
        double* gpuBuffer = getAvailableGPUBuffer();
        _dataMapCUDA[node] = RooSpan<const double>(gpuBuffer, _nEvents);
        rbc::dispatch_gpu->memcpyToGPU(gpuBuffer, gpuBuffer, _nEvents*sizeof(double), info.stream);
        _dataMapCUDA[node] = RooSpan<const double>(gpuBuffer, _nEvents);
        rbc::dispatch_gpu->memcpyToGPU(gpuBuffer, _dataMapCPU[node].data(), _nEvents*sizeof(double), info.stream);
        rbc::dispatch_gpu->cudaEventRecord(info.event, info.stream);
      }
    }
    updateMyClients(node);
    updateMyServers(node);
  } // while (nNodes)
  
  // recycle the top node's buffer and return the final value
  if (_nodeInfos.at(&_topNode).computeInGPU) {
    _gpuBuffers.push( const_cast<double*>( _dataMapCUDA[&_topNode].data() ));
    rbc::dispatch = rbc::dispatch_gpu;
    return _topNode.reduce(_dataMapCUDA[&_topNode].data(), _nEvents);
  }
  else {
    rbc::dispatch = rbc::dispatch_cpu;
    _cpuBuffers.push( const_cast<double*>( _dataMapCPU[&_topNode].data() ));
    return _topNode.reduce(_dataMapCPU[&_topNode].data(), _nEvents);
  }
}

// TODO: Put integrals seperately in the computation queue
// For now, we just assume they are scalar and assign them some temporary memory
void RooFitDriver::handleIntegral(const RooAbsReal* node)
{
  if (auto pAbsPdf = dynamic_cast<const RooAbsPdf*>(node))
  {
    auto integral = pAbsPdf->getIntegral(*_data->get());
    _nonDerivedValues.push_back(integral->getVal());
    _dataMapCPU[integral] = _dataMapCUDA[integral] = RooSpan<const double>(&_nonDerivedValues.back(),1);
  }
}

void RooFitDriver::assignToGPU(const RooAbsReal* node)
{
  NodeInfo& info = _nodeInfos.at(node);
  info.remServers=-1;
  // wait for every server to finish
  for (auto* server : node->servers())
  {
    auto pServer = static_cast<const RooAbsReal*>(server);
    if (_nodeInfos.count(pServer)==0) continue;
    const auto& infoServer = _nodeInfos.at(pServer);
    if (infoServer.event)
      rbc::dispatch_gpu->cudaStreamWaitEvent(info.stream, infoServer.event);
  }
  double* buffer = getAvailableGPUBuffer();
  _dataMapCUDA[node] = RooSpan<const double>(buffer, _nEvents);
  handleIntegral(node);
  rbc::dispatch = rbc::dispatch_gpu;
  node->computeBatch(buffer, _nEvents, _dataMapCUDA);
  rbc::dispatch = nullptr;
  rbc::dispatch_gpu->cudaEventRecord(info.event, info.stream);
  if (info.copyAfterEvaluation)
  {
    double* pinnedBuffer = getAvailablePinnedBuffer();
    rbc::dispatch_gpu->memcpyToCPU(pinnedBuffer, buffer, _nEvents*sizeof(double), info.stream);
    _dataMapCPU[node] = RooSpan<const double>(pinnedBuffer, _nEvents);
  }
  updateMyClients(node);
}

void RooFitDriver::updateMyClients(const RooAbsReal* node)
{
  NodeInfo& info = _nodeInfos.at(node);
  for (auto* client : node->valueClients())
  {
    auto pClient = static_cast<const RooAbsReal*>(client);
    if (_nodeInfos.count(pClient)==0) continue; //client not part of the computation graph
    NodeInfo& infoClient = _nodeInfos.at(pClient);
    
    if (info.remServers==-1 && infoClient.computeInGPU && --infoClient.remServers==0) 
      assignToGPU(pClient); // updateMyCilents called when assigning to gpu
    else if (info.remServers==-2 && info.computeInGPU && !infoClient.computeInGPU) 
      --infoClient.remServers; // updateMyClients called after finishing a gpu node
    else if (!info.computeInGPU && --infoClient.remServers==0 && infoClient.computeInGPU) 
      assignToGPU(pClient); // updateMyClients called after finishing a cpu node
  }
}

void RooFitDriver::updateMyServers(const RooAbsReal* node)
{
  for (auto* server : node->servers())
  {
    auto pServer = static_cast<const RooAbsReal*>(server);
    if (_nodeInfos.count(pServer)==0) continue;
    auto& info = _nodeInfos.at(pServer);
    if (--info.remClients>0) continue;
    if (info.computeInScalarMode) continue;
    if (info.copyAfterEvaluation)
    {
      _gpuBuffers.push( const_cast<double*>( _dataMapCUDA[pServer].data() ));
      _pinnedBuffers.push( const_cast<double*>( _dataMapCPU[pServer].data() ));
    }
    else if (info.computeInGPU)
      _gpuBuffers.push( const_cast<double*>( _dataMapCUDA[pServer].data() ));
    else
      _cpuBuffers.push( const_cast<double*>( _dataMapCPU[pServer].data() ));
  }
}

void RooFitDriver::markGPUNodes()
{
  for (auto& item:_nodeInfos)
    item.second.computeInGPU = item.first->canComputeBatchWithCuda();

  for (auto& item:_nodeInfos)
  {
    if (item.second.computeInScalarMode) continue; // scalar nodes don't need copying
    for (auto* client : static_range_cast<const RooAbsReal*>(item.first->valueClients()))
      if (_nodeInfos.count(client) > 0 && item.second.computeInGPU != _nodeInfos.at(client).computeInGPU)
      {
        item.second.copyAfterEvaluation = true;
        break;
      }
  }

  for (auto& item:_nodeInfos)
    if (item.second.computeInGPU || item.second.copyAfterEvaluation)
    {
      item.second.event = rbc::dispatch_gpu->newCudaEvent(false);
      item.second.stream = rbc::dispatch_gpu->newCudaStream();
    }
}
