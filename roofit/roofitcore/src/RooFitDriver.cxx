/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *   Emmanouil Michalainas, CERN 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\file RooFitDriver.cxx
\class RooFitDriver
\ingroup Roofitcore

This class is responsible for evaluating a RooAbsReal object. Currently,
it is being used for evaluating a RooNLLVarNew object and supplying the
value to the minimizer, during a fit. The class scans the dependencies and
schedules the computations in a secure and efficient way. The computations
take place in the RooBatchCompute library and can be carried off by
either the cpu or a cuda gpu; this class takes care of data transfers.
An instance of this class is created every time RooAbsPdf::fitTo() is called
and gets destroyed when the fitting ends.
**/

#include "RooFitDriver.h"
#include "RooAbsData.h"
#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooBatchCompute.h"
#include "RooNLLVarNew.h"
#include "RooRealVar.h"
#include "RunContext.h"
#include "RooDataSet.h"

#include <thread>

namespace ROOT {
namespace Experimental {

/**
Construct a new RooFitDriver. The constructor analyzes and saves metadata about the graph,
useful for the evaluation of it that will be done later. In case cuda mode is selected,
there's also some cuda-related initialization.

\param data The dataset for the fitting model
\param topNode The RooNLLVaNew object that sits on top of the graph and whose value the minimiser needs.
\param batchMode The computation mode of the RooBatchCompute library, accepted values are `RooBatchCompute::Cpu` and
`RooBatchCompute::Cuda`.
**/
RooFitDriver::RooFitDriver(const RooAbsData &data, const RooNLLVarNew &topNode, RooBatchCompute::BatchMode batchMode)
   : _name{topNode.GetName()}, _title{topNode.GetTitle()}, _parameters{*std::unique_ptr<RooArgSet>(
                                                              topNode.getParameters(data, true))},
     _batchMode{batchMode}, _topNode{topNode}, _data{&data}, _nEvents{static_cast<size_t>(data.numEntries())}
{
   // fill the RunContext with the observable data and map the observables
   // by namePtr in order to replace their memory addresses later, with
   // the ones from the variables that are actually in the computation graph.
   RooBatchCompute::RunContext evalData;
   data.getBatches(evalData, 0, _nEvents);
   _dataMapCPU = evalData.spans;
   std::unordered_map<const TNamed *, const RooAbsReal *> nameResolver;
   for (auto &it : _dataMapCPU)
      nameResolver[it.first->namePtr()] = it.first;

   // Check if there is a batch for weights and if it's already in the dataMap.
   // If not, we need to put the batch and give as a key a RooRealVar* that has
   // the same name as RooNLLVarNew's _weight proxy, so that it gets renamed like
   // every other observable.
   RooSpan<const double> weights = data.getWeightBatch(0, _nEvents);
   std::string weightVarName = "_weight";
   if (auto *dataSet = dynamic_cast<RooDataSet const *>(&data)) {
      if (dataSet->weightVar())
         weightVarName = dataSet->weightVar()->GetName();
   }
   RooRealVar dummy(weightVarName.c_str(), "dummy", 0.0);
   const TNamed *pTNamed = dummy.namePtr();
   if (!weights.empty() && nameResolver.count(pTNamed) == 0) {
      _dataMapCPU[&dummy] = weights;
      nameResolver[pTNamed] = &dummy;
   }

   // Get a serial list of the nodes in the computation graph.
   // treeNodeServelList() is recursive and adds the top node before the children,
   // so reversing the list gives us a topological ordering of the graph.
   RooArgList list;
   _topNode.treeNodeServerList(&list);
   for (int i = list.size() - 1; i >= 0; i--) {
      auto pAbsReal = dynamic_cast<RooAbsReal *>(&list[i]);
      if (!pAbsReal)
         continue;
      const bool alreadyExists = nameResolver.count(pAbsReal->namePtr());
      const RooAbsReal *pClone = nameResolver[pAbsReal->namePtr()];
      if (alreadyExists && !pClone)
         continue; // node included multiple times in the list

      if (pClone) // this node is an observable, update the RunContext and don't add it in `_nodeInfos`.
      {
         auto it = _dataMapCPU.find(pClone);
         _dataMapCPU[pAbsReal] = it->second;
         _dataMapCPU.erase(it);

         // set nameResolver to nullptr to be able to detect future duplicates
         nameResolver[pAbsReal->namePtr()] = nullptr;
      } else // this node needs evaluation, mark it's clients
      {
         // If the node doesn't depend on any observables, there is no need to
         // loop over events and we don't need to use the batched evaluation.
         RooArgSet observablesForNode;
         pAbsReal->getObservables(_data->get(), observablesForNode);
         _nodeInfos[pAbsReal].computeInScalarMode = observablesForNode.empty() || !pAbsReal->isDerived();

         auto clients = pAbsReal->valueClients();
         for (auto *client : clients) {
            if (list.find(*client)) {
               auto pClient = static_cast<const RooAbsReal *>(client);
               ++_nodeInfos[pClient].nServers;
               ++_nodeInfos[pAbsReal].nClients;
            }
         }
      }
   }

   // Extra steps for initializing in cuda mode
   if (_batchMode != RooBatchCompute::BatchMode::Cuda)
      return;
   if (!RooBatchCompute::dispatchCUDA)
      throw std::runtime_error(std::string("In: ") + __func__ + "(), " + __FILE__ + ":" + __LINE__ +
                               ": Cuda implementation of the computing library is not available\n");

   // copy observable data to the gpu
   _cudaMemDataset =
      static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMalloc(_nEvents * _dataMapCPU.size() * sizeof(double)));
   size_t idx = 0;
   for (auto &record : _dataMapCPU) {
      _dataMapCUDA[record.first] = RooSpan<double>(_cudaMemDataset + idx, _nEvents);
      RooBatchCompute::dispatchCUDA->memcpyToCUDA(_cudaMemDataset + idx, record.second.data(),
                                                  _nEvents * sizeof(double));
      idx += _nEvents;
   }

   // create events and streams for every node
   for (auto &item : _nodeInfos) {
      item.second.event = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
      item.second.eventStart = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
      item.second.stream = RooBatchCompute::dispatchCUDA->newCudaStream();
   }
}

namespace {

template <typename T, typename Func_t>
T getAvailable(std::queue<T> &q, Func_t creator)
{
   if (q.empty())
      return static_cast<T>(creator());
   else {
      T ret = q.front();
      q.pop();
      return ret;
   }
}

template <typename T, typename Func_t>
void clearQueue(std::queue<T> &q, Func_t destroyer)
{
   while (!q.empty()) {
      destroyer(q.front());
      q.pop();
   }
}
} // end anonymous namespace

RooFitDriver::~RooFitDriver()
{
   clearQueue(_cpuBuffers, [](double *ptr) { delete[] ptr; });
   if (_batchMode == RooBatchCompute::BatchMode::Cuda) {
      clearQueue(_gpuBuffers, [](double *ptr) { RooBatchCompute::dispatchCUDA->cudaFree(ptr); });
      clearQueue(_pinnedBuffers, [](double *ptr) { RooBatchCompute::dispatchCUDA->cudaFreeHost(ptr); });
      RooBatchCompute::dispatchCUDA->cudaFree(_cudaMemDataset);
   }
}

double *RooFitDriver::getAvailableCPUBuffer()
{
   return getAvailable(_cpuBuffers, [=]() { return new double[_nEvents]; });
}
double *RooFitDriver::getAvailableGPUBuffer()
{
   return getAvailable(_gpuBuffers,
                       [=]() { return RooBatchCompute::dispatchCUDA->cudaMalloc(_nEvents * sizeof(double)); });
}
double *RooFitDriver::getAvailablePinnedBuffer()
{
   return getAvailable(_pinnedBuffers,
                       [=]() { return RooBatchCompute::dispatchCUDA->cudaMallocHost(_nEvents * sizeof(double)); });
}

/// Returns the value of the top node in the computation graph
double RooFitDriver::getVal()
{
   // In a cuda fit, use first 3 fits to determine the execution times
   // and the hardware that computes each part of the graph
   if (_batchMode == RooBatchCompute::BatchMode::Cuda && ++_getValInvocations <= 3)
      markGPUNodes();

   for (auto &item : _nodeInfos) {
      item.second.remClients = item.second.nClients;
      item.second.remServers = item.second.nServers;
   }
   _nonDerivedValues.clear();
   _nonDerivedValues.reserve(_nodeInfos.size()); // to avoid reallocation

   // find initial gpu nodes and assign them to gpu
   for (const auto &it : _nodeInfos)
      if (it.second.remServers == 0 && it.second.computeInGPU)
         assignToGPU(it.first);

   int nNodes = _nodeInfos.size();
   while (nNodes) {
      // find finished gpu nodes
      if (_batchMode == RooBatchCompute::BatchMode::Cuda)
         for (auto &it : _nodeInfos)
            if (it.second.remServers == -1 && !RooBatchCompute::dispatchCUDA->streamIsActive(it.second.stream)) {
               if (_getValInvocations == 2) {
                  float ms = RooBatchCompute::dispatchCUDA->cudaEventElapsedTime(it.second.eventStart, it.second.event);
                  it.second.cudaTime += std::chrono::microseconds{int(1000.0 * ms)};
               }
               it.second.remServers = -2;
               nNodes--;
               updateMyClients(it.first);
               updateMyServers(it.first);
            }

      // find next cpu node
      auto it = _nodeInfos.begin();
      for (; it != _nodeInfos.end(); it++)
         if (it->second.remServers == 0 && !it->second.computeInGPU)
            break;

      // if no cpu node available sleep for a while to save cpu usage
      if (it == _nodeInfos.end()) {
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
         continue;
      }

      // compute next cpu node
      const RooAbsReal *node = it->first;
      NodeInfo &info = it->second;
      info.remServers = -2; // so that it doesn't get picked again
      nNodes--;
      if (info.computeInScalarMode) {
         _nonDerivedValues.push_back(node->getVal(_data->get()));
         _dataMapCPU[node] = _dataMapCUDA[node] = RooSpan<const double>(&_nonDerivedValues.back(), 1);
      } else {
         double *buffer = info.copyAfterEvaluation ? getAvailablePinnedBuffer() : getAvailableCPUBuffer();
         _dataMapCPU[node] = RooSpan<const double>(buffer, _nEvents);
         handleIntegral(node);
         // compute node and measure the time the first time
         if (_getValInvocations == 1) {
            using namespace std::chrono;
            auto start = steady_clock::now();
            node->computeBatch(RooBatchCompute::dispatchCPU, buffer, _nEvents, _dataMapCPU);
            info.cpuTime = duration_cast<microseconds>(steady_clock::now() - start);
         } else
            node->computeBatch(RooBatchCompute::dispatchCPU, buffer, _nEvents, _dataMapCPU);
         if (info.copyAfterEvaluation) {
            double *gpuBuffer = getAvailableGPUBuffer();
            _dataMapCUDA[node] = RooSpan<const double>(gpuBuffer, _nEvents);
            RooBatchCompute::dispatchCUDA->memcpyToCUDA(gpuBuffer, buffer, _nEvents * sizeof(double), info.stream);
            RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
         }
      }
      updateMyClients(node);
      updateMyServers(node);
   } // while (nNodes)

   // recycle the top node's buffer and return the final value
   if (_nodeInfos.at(&_topNode).computeInGPU) {
      _gpuBuffers.push(const_cast<double *>(_dataMapCUDA[&_topNode].data()));
      return _topNode.reduce(RooBatchCompute::dispatchCUDA, _dataMapCUDA[&_topNode].data(), _nEvents);
   } else {
      _cpuBuffers.push(const_cast<double *>(_dataMapCPU[&_topNode].data()));
      return _topNode.reduce(RooBatchCompute::dispatchCPU, _dataMapCPU[&_topNode].data(), _nEvents);
   }
}

/// Handles the computation of the integral of a PDF for normalization purposes,
/// before the pdf is computed.
void RooFitDriver::handleIntegral(const RooAbsReal *node)
{
   // TODO: Put integrals seperately in the computation queue
   // For now, we just assume they are scalar and assign them some temporary memory
   if (auto pAbsPdf = dynamic_cast<const RooAbsPdf *>(node)) {
      auto integral = pAbsPdf->getIntegral(*_data->get());
      _nonDerivedValues.push_back(integral->getVal());
      _dataMapCPU[integral] = _dataMapCUDA[integral] = RooSpan<const double>(&_nonDerivedValues.back(), 1);
   }
}

/// Assign a node to be computed in the GPU. Scan it's clients and also assign them
/// in case they only depend on gpu nodes.
void RooFitDriver::assignToGPU(const RooAbsReal *node)
{
   NodeInfo &info = _nodeInfos.at(node);
   info.remServers = -1;
   // wait for every server to finish
   for (auto *server : node->servers()) {
      auto pServer = static_cast<const RooAbsReal *>(server);
      if (_nodeInfos.count(pServer) == 0)
         continue;
      const auto &infoServer = _nodeInfos.at(pServer);
      if (infoServer.event)
         RooBatchCompute::dispatchCUDA->cudaStreamWaitEvent(info.stream, infoServer.event);
   }
   double *buffer = getAvailableGPUBuffer();
   _dataMapCUDA[node] = RooSpan<const double>(buffer, _nEvents);
   handleIntegral(node);
   // measure launching overhead (add computation time later)
   if (_getValInvocations == 2) {
      using namespace std::chrono;
      RooBatchCompute::dispatchCUDA->cudaEventRecord(info.eventStart, info.stream);
      auto start = steady_clock::now();
      node->computeBatch(RooBatchCompute::dispatchCUDA, buffer, _nEvents, _dataMapCUDA);
      info.cudaTime = duration_cast<microseconds>(steady_clock::now() - start);
   } else
      node->computeBatch(RooBatchCompute::dispatchCUDA, buffer, _nEvents, _dataMapCUDA);
   RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
   if (info.copyAfterEvaluation) {
      double *pinnedBuffer = getAvailablePinnedBuffer();
      RooBatchCompute::dispatchCUDA->memcpyToCPU(pinnedBuffer, buffer, _nEvents * sizeof(double), info.stream);
      _dataMapCPU[node] = RooSpan<const double>(pinnedBuffer, _nEvents);
   }
   updateMyClients(node);
}

/// Check the clients of a node that has been computed and assign them to gpu
/// if they are ready to be computed
void RooFitDriver::updateMyClients(const RooAbsReal *node)
{
   NodeInfo &info = _nodeInfos.at(node);
   for (auto *client : node->valueClients()) {
      auto pClient = static_cast<const RooAbsReal *>(client);
      if (_nodeInfos.count(pClient) == 0)
         continue; // client not part of the computation graph
      NodeInfo &infoClient = _nodeInfos.at(pClient);

      if (info.remServers == -1 && infoClient.computeInGPU && --infoClient.remServers == 0)
         assignToGPU(pClient); // updateMyCilents called when assigning to gpu
      else if (info.remServers == -2 && info.computeInGPU && !infoClient.computeInGPU)
         --infoClient.remServers; // updateMyClients called after finishing a gpu node
      else if (!info.computeInGPU && --infoClient.remServers == 0 && infoClient.computeInGPU)
         assignToGPU(pClient); // updateMyClients called after finishing a cpu node
   }
}

/// Check the servers of a node that has been computed and release it's resources
/// if they are no longer needed
void RooFitDriver::updateMyServers(const RooAbsReal *node)
{
   for (auto *server : node->servers()) {
      auto pServer = static_cast<const RooAbsReal *>(server);
      if (_nodeInfos.count(pServer) == 0)
         continue;
      auto &info = _nodeInfos.at(pServer);
      if (--info.remClients > 0)
         continue;
      if (info.computeInScalarMode)
         continue;
      if (info.copyAfterEvaluation) {
         _gpuBuffers.push(const_cast<double *>(_dataMapCUDA[pServer].data()));
         _pinnedBuffers.push(const_cast<double *>(_dataMapCPU[pServer].data()));
      } else if (info.computeInGPU)
         _gpuBuffers.push(const_cast<double *>(_dataMapCUDA[pServer].data()));
      else
         _cpuBuffers.push(const_cast<double *>(_dataMapCPU[pServer].data()));
   }
}

/// Measure the time for transfering data from host to device and vice-versa
std::pair<std::chrono::microseconds, std::chrono::microseconds> RooFitDriver::memcpyBenchmark()
{
   using namespace std::chrono;
   std::pair<microseconds, microseconds> ret;
   auto hostArr = static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMallocHost(_nEvents * sizeof(double)));
   auto deviArr = static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMalloc(_nEvents * sizeof(double)));
   for (int i = 0; i < 5; i++) {
      auto start = steady_clock::now();
      RooBatchCompute::dispatchCUDA->memcpyToCUDA(deviArr, hostArr, _nEvents * sizeof(double));
      ret.first += duration_cast<microseconds>(steady_clock::now() - start);
      start = steady_clock::now();
      RooBatchCompute::dispatchCUDA->memcpyToCPU(hostArr, deviArr, _nEvents * sizeof(double));
      ret.second += duration_cast<microseconds>(steady_clock::now() - start);
   }
   RooBatchCompute::dispatchCUDA->cudaFreeHost(hostArr);
   RooBatchCompute::dispatchCUDA->cudaFree(deviArr);
   ret.first /= 5;
   ret.second /= 5;
   return ret;
}

/// Decides which nodes are assigned to the gpu in a cuda fit. In the 1st iteration,
/// everything is computed in cpu for measuring the cpu time. In the 2nd iteration,
/// everything is computed in gpu (if possible) to measure the gpu time.
/// In the 3rd iteration, this methods simulates the computation of the whole graph
/// and the time it takes and decides what to compute in gpu. The decision is made on the
/// basis avoiding leaving either the gpu or the cpu idle at any time, if possible,
/// and on assigning to each piece of hardware a computation that is significantly slower on the
/// other part.
void RooFitDriver::markGPUNodes()
{
   if (_getValInvocations == 1)
      return;                        // leave everything to be computed (and timed) in cpu
   else if (_getValInvocations == 2) // compute (and time) as much as possible in gpu
      for (auto &item : _nodeInfos)
         item.second.computeInGPU = !item.second.computeInScalarMode && item.first->canComputeBatchWithCuda();
   else // assign nodes to gpu using a greedy algorithm
   {
      // deletion of the timing events (to be replaced later by non-timing events)
      for (auto &item : _nodeInfos) {
         item.second.computeInGPU = item.second.copyAfterEvaluation = false;
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(item.second.event);
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(item.second.eventStart);
         item.second.event = item.second.eventStart = nullptr;
      }

      using namespace std::chrono;
      auto transferTimes = memcpyBenchmark();
      microseconds h2dTime = transferTimes.first;
      microseconds d2hTime = transferTimes.second;

      const RooAbsReal *cpuNode = nullptr, *cudaNode = nullptr;
      microseconds simulatedTime{0};
      int nNodes = _nodeInfos.size();
      // launch scalar nodes (assume they are computed in 0 time)
      for (auto &it : _nodeInfos)
         if (it.second.computeInScalarMode) {
            nNodes--;
            it.second.timeLaunched = microseconds{0};
         }

      while (nNodes) {
         microseconds minDiff = microseconds::max(), maxDiff = -minDiff; // diff = cpuTime - cudaTime
         const RooAbsReal *cpuCandidate = nullptr, *cudaCandidate = nullptr;
         microseconds cpuDelay, cudaDelay;
         for (auto &it : _nodeInfos) {
            if (it.second.timeLaunched >= microseconds{0})
               continue; // already launched
            microseconds diff{it.second.cpuTime - it.second.cudaTime}, cpuWait{0}, cudaWait{0};

            for (auto *server : it.first->servers()) {
               auto pServer = static_cast<const RooAbsReal *>(server);
               if (_nodeInfos.count(pServer) == 0)
                  continue;
               auto &info = _nodeInfos.at(pServer);

               // dependencies not computed yet
               if (info.timeLaunched < microseconds{0})
                  goto nextCandidate;
               if (info.computeInGPU)
                  cpuWait = std::max(cpuWait, info.timeLaunched + info.cudaTime + d2hTime - simulatedTime);
               else
                  cudaWait = std::max(cudaWait, info.timeLaunched + info.cpuTime + h2dTime - simulatedTime);
            }

            diff += cpuWait - cudaWait;
            if (diff < minDiff) {
               minDiff = diff;
               cpuDelay = cpuWait;
               cpuCandidate = it.first;
            }
            if (diff > maxDiff && it.first->canComputeBatchWithCuda()) {
               maxDiff = diff;
               cudaDelay = cudaWait;
               cudaCandidate = it.first;
            }
         nextCandidate:;
         } // for (auto& it:_nodeInfos)

         // don't compute same node twice
         if (cpuCandidate == cudaCandidate && !cpuNode && !cudaNode) {
            if (minDiff < microseconds{0})
               cudaCandidate = nullptr;
            else
               cpuCandidate = nullptr;
         }
         if (cpuCandidate && !cpuNode) {
            cpuNode = cpuCandidate;
            _nodeInfos.at(cpuNode).timeLaunched = simulatedTime + cpuDelay;
            nNodes--;
         }
         if (cudaCandidate && !cudaNode) {
            cudaNode = cudaCandidate;
            _nodeInfos.at(cudaNode).timeLaunched = simulatedTime + cudaDelay;
            _nodeInfos.at(cudaNode).computeInGPU = true;
            nNodes--;
         }

         microseconds etaCPU{microseconds::max()}, etaCUDA{microseconds::max()};
         if (cpuNode)
            etaCPU = _nodeInfos[cpuNode].timeLaunched + _nodeInfos[cpuNode].cpuTime;
         if (cudaNode)
            etaCUDA = _nodeInfos[cudaNode].timeLaunched + _nodeInfos[cudaNode].cudaTime;
         simulatedTime = std::min(etaCPU, etaCUDA);
         if (etaCPU < etaCUDA)
            cpuNode = nullptr;
         else
            cudaNode = nullptr;
      } // while(nNodes)
   }    // else (_getValInvocations > 2)

   for (auto &item : _nodeInfos)
      if (!item.second.computeInScalarMode) // scalar nodes don't need copying
         for (auto *client : item.first->valueClients()) {
            auto pClient = static_cast<const RooAbsReal *>(client);
            if (_nodeInfos.count(pClient) == 0)
               continue;
            auto &info = _nodeInfos.at(pClient);
            if (item.second.computeInGPU != info.computeInGPU) {
               item.second.copyAfterEvaluation = true;
               break;
            }
         }

   // restore a cudaEventDisableTiming event when necessary
   if (_getValInvocations == 3)
      for (auto &item : _nodeInfos)
         if (item.second.computeInGPU || item.second.copyAfterEvaluation)
            item.second.event = RooBatchCompute::dispatchCUDA->newCudaEvent(false);
}

} // namespace Experimental
} // namespace ROOT
