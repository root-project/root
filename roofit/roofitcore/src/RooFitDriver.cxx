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

This class can evaluate a RooAbsReal object in other ways than recursive graph
traversal. Currently, it is being used for evaluating a RooAbsReal object and
supplying the value to the minimizer, during a fit. The class scans the
dependencies and schedules the computations in a secure and efficient way. The
computations take place in the RooBatchCompute library and can be carried off
by either the CPU or a CUDA-supporting GPU. The RooFitDriver class takes care
of data transfers. An instance of this class is created every time
RooAbsPdf::fitTo() is called and gets destroyed when the fitting ends.
**/

#include <RooFitDriver.h>

#include <RooAbsCategory.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooAbsPdf.h>
#include <RooArgList.h>
#include <RooBatchCompute.h>
#include <RooMsgService.h>
#include <RooBatchCompute/Initialisation.h>
#include <RooBatchCompute/DataKey.h>
#include <RooFit/BatchModeDataHelpers.h>
#include <RooFit/BatchModeHelpers.h>
#include <RooFit/CUDAHelpers.h>
#include <RooFit/Detail/Buffers.h>

#include <iomanip>
#include <numeric>
#include <thread>
#include <unordered_set>

namespace ROOT {
namespace Experimental {

/// A struct used by the RooFitDriver to store information on the RooAbsArgs in
/// the computation graph.
struct NodeInfo {

   NodeInfo() {}

   // No copying because of the owned CUDA pointers and buffers
   NodeInfo(const NodeInfo &) = delete;
   NodeInfo &operator=(const NodeInfo &) = delete;

   /// Check the servers of a node that has been computed and release it's resources
   /// if they are no longer needed.
   void decrementRemainingClients()
   {
      if (--remClients == 0) {
         buffer.reset();
      }
   }

   RooAbsArg *absArg = nullptr;

   std::unique_ptr<Detail::AbsBuffer> buffer;

   cudaEvent_t *event = nullptr;
   cudaEvent_t *eventStart = nullptr;
   cudaStream_t *stream = nullptr;
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
   bool fromDataset = false;
   std::size_t outputSize = 1;
   ~NodeInfo()
   {
      if (event)
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(event);
      if (eventStart)
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(eventStart);
      if (stream)
         RooBatchCompute::dispatchCUDA->deleteCudaStream(stream);
   }
};

/// Construct a new RooFitDriver. The constructor analyzes and saves metadata about the graph,
/// useful for the evaluation of it that will be done later. In case the CUDA mode is selected,
/// there's also some CUDA-related initialization.
///
/// \param[in] topNode The RooAbsReal object that sits on top of the
///            computation graph that we want to evaluate.
/// \param[in] normSet Normalization set for the evaluation
/// \param[in] batchMode The computation mode, accepted values are
///            `RooBatchCompute::Cpu` and `RooBatchCompute::Cuda`.
RooFitDriver::RooFitDriver(const RooAbsReal &topNode, RooArgSet const &normSet, RooFit::BatchModeOption batchMode)
   : _batchMode{batchMode}, _topNode{topNode}, _normSet{std::make_unique<RooArgSet>(normSet)}
{
   // Initialize RooBatchCompute
   RooBatchCompute::init();

   // Some checks and logging of used architectures
   RooFit::BatchModeHelpers::logArchitectureInfo(_batchMode);

   // Get a serial list of the nodes in the computation graph.
   // treeNodeServelList() is recursive and adds the top node before the children,
   // so reversing the list gives us a topological ordering of the graph.
   RooArgList serverList;
   _topNode.treeNodeServerList(&serverList, nullptr, true, true, false, true);

   // To remove duplicates via the RooArgSet deduplication, we have to fill the
   // set in reverse order because that's the dependency ordering of the graph.
   RooArgSet serverSet;
   std::unordered_map<TNamed const *, RooAbsArg *> instanceMap;
   for (std::size_t iNode = serverList.size(); iNode > 0; --iNode) {
      RooAbsArg *arg = &serverList[iNode - 1];
      if (instanceMap.find(arg->namePtr()) != instanceMap.end()) {
         if (arg != instanceMap.at(arg->namePtr())) {
            serverSet.remove(*instanceMap.at(arg->namePtr()));
         }
      }
      instanceMap[arg->namePtr()] = arg;
      serverSet.add(*arg);
   }

   // Fill the ordered nodes list and initialize the node info structs.
   for (RooAbsArg *arg : serverSet) {
      _orderedNodes.add(*arg);
      _nodeInfos[arg];
   }

   if (_batchMode == RooFit::BatchModeOption::Cuda) {
      // For the CUDA mode, we need to keep track of the number of servers and
      // clients of each node.
      for (RooAbsArg *arg : _orderedNodes) {
         auto &argInfo = _nodeInfos.at(arg);
         argInfo.absArg = arg;

         for (auto *client : arg->clients()) {
            // we use containsInstance instead of find to match by pointer and not name
            if (!serverSet.containsInstance(*client))
               continue;

            auto &clientInfo = _nodeInfos.at(client);

            ++clientInfo.nServers;
            ++argInfo.nClients;
         }
      }

      // create events and streams for every node
      for (auto &item : _nodeInfos) {
         item.second.event = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
         item.second.eventStart = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
         item.second.stream = RooBatchCompute::dispatchCUDA->newCudaStream();
      }
   }
}

void RooFitDriver::setData(RooAbsData const &data, std::string_view rangeName,
                           RooAbsCategory const *indexCatForSplitting)
{
   setData(RooFit::BatchModeDataHelpers::getDataSpans(data, rangeName, indexCatForSplitting, _vectorBuffers));
}

void RooFitDriver::setData(DataSpansMap const &dataSpans)
{
   if (!_dataMapCPU.empty() && _dataMapCUDA.empty()) {
      throw std::runtime_error("You can call RooFitDriver::setData() only on a freshly-constructed RooFitDriver!");
   }

   // Iterate over the given data spans and add them to the data map. Check if
   // they are used in the computation graph. If yes, add the span to the data
   // map and set the node info accordingly.
   for (auto const &span : dataSpans) {
      using RooFit::BatchModeHelpers::NamePtrWrapper;
      NamePtrWrapper dataKey{span.first};
      _dataMapCPU[dataKey] = span.second;
      auto found = _nodeInfos.find(dataKey);
      if (found != _nodeInfos.end()) {
         auto &argInfo = found->second;
         argInfo.outputSize = span.second.size();
         argInfo.fromDataset = true;
      }
   }

   determineOutputSizes();

   for (auto *arg : _orderedNodes) {
      auto &info = _nodeInfos.at(arg);

      // If the node evaluation doesn't involve a loop over entries, we can
      // always use the scalar mode.
      info.computeInScalarMode = info.outputSize == 1 && !arg->isReducerNode();

      // We don't need dirty flag propagation for nodes evaluated in batch
      // mode, because the driver takes care of deciding which node needs to be
      // re-evaluated. However, dirty flag propagation must be kept for reducer
      // nodes, because their clients are evaluated in scalar mode.
      if (!info.computeInScalarMode && !arg->isReducerNode()) {
         setOperMode(arg, RooAbsArg::ADirty);
      }
   }

   // Extra steps for initializing in cuda mode
   if (_batchMode != RooFit::BatchModeOption::Cuda)
      return;

   std::size_t totalSize = 0;
   for (auto &record : _dataMapCPU) {
      totalSize += record.second.size();
   }
   // copy observable data to the GPU
   // TODO: use separate buffers here
   _cudaMemDataset = static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMalloc(totalSize * sizeof(double)));
   size_t idx = 0;
   for (auto &record : _dataMapCPU) {
      std::size_t size = record.second.size();
      _dataMapCUDA[record.first] = RooSpan<double>(_cudaMemDataset + idx, size);
      RooBatchCompute::dispatchCUDA->memcpyToCUDA(_cudaMemDataset + idx, record.second.data(), size * sizeof(double));
      idx += size;
   }
}

RooFitDriver::~RooFitDriver()
{
   if (_batchMode == RooFit::BatchModeOption::Cuda) {
      RooBatchCompute::dispatchCUDA->cudaFree(_cudaMemDataset);
   }
}

std::vector<double> RooFitDriver::getValues()
{
   getVal();
   auto const &nodeInfo = _nodeInfos.at(&_topNode);
   if (nodeInfo.computeInGPU) {
      std::size_t nOut = nodeInfo.outputSize;
      std::vector<double> out(nOut);
      RooBatchCompute::dispatchCUDA->memcpyToCPU(out.data(), _dataMapCPU.at(&_topNode).data(), nOut * sizeof(double));
      _dataMapCPU[&_topNode] = RooSpan<const double>(out.data(), nOut);
      return out;
   }
   // We copy the data to the output vector
   auto dataSpan = _dataMapCPU.at(&_topNode);
   std::vector<double> out;
   out.reserve(dataSpan.size());
   for (auto const &x : dataSpan) {
      out.push_back(x);
   }
   return out;
}

void RooFitDriver::computeCPUNode(const RooAbsArg *node, NodeInfo &info)
{
   using namespace Detail;

   auto nodeAbsReal = dynamic_cast<RooAbsReal const *>(node);
   auto nodeAbsCategory = dynamic_cast<RooAbsCategory const *>(node);
   assert(nodeAbsReal || nodeAbsCategory);

   const std::size_t nOut = info.outputSize;

   if (info.computeInScalarMode) {
      // compute in scalar mode
      _nonDerivedValues.push_back(nodeAbsCategory ? nodeAbsCategory->getIndex() : nodeAbsReal->getVal(_normSet.get()));
      _dataMapCPU[node] = _dataMapCUDA[node] = RooSpan<const double>(&_nonDerivedValues.back(), nOut);
   } else if (nOut == 1) {
      handleIntegral(node);
      _nonDerivedValues.push_back(0.0);
      _dataMapCPU[node] = _dataMapCUDA[node] = RooSpan<const double>(&_nonDerivedValues.back(), nOut);
      nodeAbsReal->computeBatch(nullptr, &_nonDerivedValues.back(), nOut, _dataMapCPU);
   } else {
      handleIntegral(node);
      info.buffer = info.copyAfterEvaluation ? makePinnedBuffer(nOut, info.stream) : makeCpuBuffer(nOut);
      double *buffer = info.buffer->cpuWritePtr();
      _dataMapCPU[node] = RooSpan<const double>(buffer, nOut);
      // compute node and measure the time the first time
      if (_getValInvocations == 1) {
         using namespace std::chrono;
         auto start = steady_clock::now();
         nodeAbsReal->computeBatch(nullptr, buffer, nOut, _dataMapCPU);
         info.cpuTime = duration_cast<microseconds>(steady_clock::now() - start);
      } else {
         nodeAbsReal->computeBatch(nullptr, buffer, nOut, _dataMapCPU);
      }
      if (info.copyAfterEvaluation) {
         _dataMapCUDA[node] = RooSpan<const double>(info.buffer->gpuReadPtr(), nOut);
         if (info.event) {
            RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
         }
      }
   }
}

/// Returns the value of the top node in the computation graph
double RooFitDriver::getVal()
{
   ++_getValInvocations;

   _nonDerivedValues.clear();
   _nonDerivedValues.reserve(_orderedNodes.size()); // to avoid reallocation

   if (_batchMode == RooFit::BatchModeOption::Cuda) {
      return getValHeterogeneous();
   }

   for (std::size_t iNode = 0; iNode < _orderedNodes.size(); ++iNode) {
      RooAbsArg *node = _orderedNodes.at(iNode);
      auto &nodeInfo = _nodeInfos.at(node);
      if (!nodeInfo.fromDataset) {
         computeCPUNode(node, nodeInfo);
      }
   }
   // return the final value
   return _dataMapCPU.at(&_topNode)[0];
}

/// Returns the value of the top node in the computation graph
double RooFitDriver::getValHeterogeneous()
{
   for (auto &item : _nodeInfos) {
      item.second.remClients = item.second.nClients;
      item.second.remServers = item.second.nServers;
   }

   // In a cuda fit, use first 3 fits to determine the execution times
   // and the hardware that computes each part of the graph
   if (_batchMode == RooFit::BatchModeOption::Cuda && _getValInvocations <= 3)
      markGPUNodes();

   // find initial gpu nodes and assign them to gpu
   for (const auto &it : _nodeInfos)
      if (it.second.remServers == 0 && it.second.computeInGPU)
         assignToGPU(it.second.absArg);

   auto const &topNodeInfo = _nodeInfos.at(&_topNode);
   while (topNodeInfo.remServers != -2) {
      // find finished gpu nodes
      for (auto &it : _nodeInfos) {
         if (it.second.remServers == -1 && !RooBatchCompute::dispatchCUDA->streamIsActive(it.second.stream)) {
            if (_getValInvocations == 2) {
               float ms = RooBatchCompute::dispatchCUDA->cudaEventElapsedTime(it.second.eventStart, it.second.event);
               it.second.cudaTime += std::chrono::microseconds{int(1000.0 * ms)};
            }
            it.second.remServers = -2;
            // Decrement number of remaining servers for clients and start GPU computations
            for (auto *client : it.second.absArg->clients()) {
               // client not part of the computation graph
               if (!isInComputationGraph(client))
                  continue;
               NodeInfo &infoClient = _nodeInfos.at(client);

               --infoClient.remServers;
               if (infoClient.computeInGPU && infoClient.remServers == 0) {
                  assignToGPU(client);
               }
            }
            for (auto *server : it.second.absArg->servers()) {
               _nodeInfos.at(server).decrementRemainingClients();
            }
         }
      }

      // find next CPU node
      auto it = _nodeInfos.begin();
      for (; it != _nodeInfos.end(); it++) {
         if (it->second.remServers == 0 && !it->second.computeInGPU)
            break;
      }

      // if no CPU node available sleep for a while to save CPU usage
      if (it == _nodeInfos.end()) {
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
         continue;
      }

      // compute next CPU node
      RooAbsArg const *node = it->second.absArg;
      NodeInfo &info = it->second;
      info.remServers = -2; // so that it doesn't get picked again

      if (!info.fromDataset) {
         computeCPUNode(node, info);
      }

      // Assign the clients that are computed on the GPU
      for (auto *client : node->clients()) {
         // client not part of the computation graph
         if (!isInComputationGraph(client))
            continue;
         NodeInfo &infoClient = _nodeInfos.at(client);
         if (--infoClient.remServers == 0 && infoClient.computeInGPU) {
            assignToGPU(client);
         }
      }
      for (auto *server : node->servers()) {
         _nodeInfos.at(server).decrementRemainingClients();
      }
   }

   // return the final value
   return _dataMapCPU.at(&_topNode)[0];
}

/// Handles the computation of the integral of a PDF for normalization purposes,
/// before the pdf is computed.
void RooFitDriver::handleIntegral(const RooAbsArg *node)
{
   // TODO: Put integrals seperately in the computation queue
   // For now, we just assume they are scalar and assign them some temporary memory
   if (auto pAbsPdf = dynamic_cast<const RooAbsPdf *>(node)) {
      auto integral = pAbsPdf->getIntegral(*_normSet);

      if (_integralInfos.count(integral) == 0) {
         auto &info = _integralInfos[integral];
         info.absArg = integral;
         for (RooAbsArg *server : integral->servers()) {
            if (server->isValueServer(*integral)) {
               info.outputSize = std::max(info.outputSize, _nodeInfos.at(server).outputSize);
            }
         }

         if (info.outputSize > 1 && _batchMode == RooFit::BatchModeOption::Cuda) {
            info.copyAfterEvaluation = true;
            info.stream = RooBatchCompute::dispatchCUDA->newCudaStream();
         } else if (info.outputSize == 1) {
            info.computeInScalarMode = true;
         }

         // We don't need dirty flag propagation for nodes evaluated by the
         // RooFitDriver, because the driver takes care of deciding which node
         // needs to be re-evaluated.
         if (!info.computeInScalarMode)
            setOperMode(integral, RooAbsArg::ADirty);
      }

      computeCPUNode(integral, _integralInfos.at(integral));
   }
}

/// Assign a node to be computed in the GPU. Scan it's clients and also assign them
/// in case they only depend on gpu nodes.
void RooFitDriver::assignToGPU(RooAbsArg const *node)
{
   using namespace Detail;

   auto nodeAbsReal = dynamic_cast<RooAbsReal const *>(node);
   assert(nodeAbsReal || dynamic_cast<RooAbsCategory const *>(node));

   NodeInfo &info = _nodeInfos.at(node);
   const std::size_t nOut = info.outputSize;

   info.remServers = -1;
   // wait for every server to finish
   for (auto *server : node->servers()) {
      if (_nodeInfos.count(server) == 0)
         continue;
      const auto &infoServer = _nodeInfos.at(server);
      if (infoServer.event)
         RooBatchCompute::dispatchCUDA->cudaStreamWaitEvent(info.stream, infoServer.event);
   }

   info.buffer = info.copyAfterEvaluation ? makePinnedBuffer(nOut, info.stream) : makeGpuBuffer(nOut);
   double *buffer = info.buffer->gpuWritePtr();
   _dataMapCUDA[node] = RooSpan<const double>(buffer, nOut);
   handleIntegral(node);
   // measure launching overhead (add computation time later)
   if (_getValInvocations == 2) {
      using namespace std::chrono;
      RooBatchCompute::dispatchCUDA->cudaEventRecord(info.eventStart, info.stream);
      auto start = steady_clock::now();
      nodeAbsReal->computeBatch(info.stream, buffer, nOut, _dataMapCUDA);
      info.cudaTime = duration_cast<microseconds>(steady_clock::now() - start);
   } else
      nodeAbsReal->computeBatch(info.stream, buffer, nOut, _dataMapCUDA);
   RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
   if (info.copyAfterEvaluation) {
      _dataMapCPU[node] = RooSpan<const double>(info.buffer->cpuReadPtr(), nOut);
   }
}

/// This methods simulates the computation of the whole graph and the time it takes
/// and decides what to compute in gpu. The decision is made on the basis of avoiding
/// leaving either the gpu or the cpu idle at any time, if possible, and on assigning
/// to each piece of hardware a computation that is significantly slower on the other part.
/// The nodes may be assigned to the non-efficient side (cpu or gpu) to prevent idleness
/// only if the absolute difference cpuTime-cudaTime does not exceed the diffThreshold.
std::chrono::microseconds RooFitDriver::simulateFit(std::chrono::microseconds h2dTime,
                                                    std::chrono::microseconds d2hTime,
                                                    std::chrono::microseconds diffThreshold)
{
   using namespace std::chrono;

   std::size_t nNodes = _nodeInfos.size();
   // launch scalar nodes (assume they are computed in 0 time)
   for (auto &it : _nodeInfos) {
      if (it.second.computeInScalarMode) {
         nNodes--;
         it.second.timeLaunched = microseconds{0};
      } else
         it.second.timeLaunched = microseconds{-1};
   }

   RooAbsArg const *cpuNode = nullptr;
   RooAbsArg const *cudaNode = nullptr;
   microseconds simulatedTime{0};
   while (nNodes) {
      microseconds minDiff = microseconds::max(), maxDiff = -minDiff; // diff = cpuTime - cudaTime
      RooAbsArg const *cpuCandidate = nullptr;
      RooAbsArg const *cudaCandidate = nullptr;
      microseconds cpuDelay, cudaDelay;
      for (auto &it : _nodeInfos) {
         RooAbsArg const *absArg = it.second.absArg;
         if (it.second.timeLaunched >= microseconds{0})
            continue; // already launched
         microseconds diff{it.second.cpuTime - it.second.cudaTime}, cpuWait{0}, cudaWait{0};

         for (auto *server : absArg->servers()) {
            if (_nodeInfos.count(server) == 0)
               continue;
            auto &info = _nodeInfos.at(server);
            if (info.computeInScalarMode)
               continue;

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
            cpuCandidate = absArg;
         }
         if (diff > maxDiff && absArg->canComputeBatchWithCuda()) {
            maxDiff = diff;
            cudaDelay = cudaWait;
            cudaCandidate = absArg;
         }
      nextCandidate:;
      } // for (auto& it:_nodeInfos)

      auto calcDiff = [&](const RooAbsArg *node) { return _nodeInfos.at(node).cpuTime - _nodeInfos.at(node).cudaTime; };
      if (cpuCandidate && calcDiff(cpuCandidate) > diffThreshold)
         cpuCandidate = nullptr;
      if (cudaCandidate && -calcDiff(cudaCandidate) > diffThreshold)
         cudaCandidate = nullptr;
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
         _nodeInfos.at(cpuNode).computeInGPU = false;
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
         etaCPU = _nodeInfos.at(cpuNode).timeLaunched + _nodeInfos.at(cpuNode).cpuTime;
      if (cudaNode)
         etaCUDA = _nodeInfos.at(cudaNode).timeLaunched + _nodeInfos.at(cudaNode).cudaTime;
      simulatedTime = std::min(etaCPU, etaCUDA);
      if (etaCPU < etaCUDA)
         cpuNode = nullptr;
      else
         cudaNode = nullptr;
   } // while(nNodes)
   return simulatedTime;
}

/// Decides which nodes are assigned to the gpu in a cuda fit. In the 1st iteration,
/// everything is computed in cpu for measuring the cpu time. In the 2nd iteration,
/// everything is computed in gpu (if possible) to measure the gpu time.
/// In the 3rd iteration, simulate the computation of the graph by calling simulateFit
/// with every distinct threshold found as timeDiff within the nodes of the graph and select
/// the best configuration. In the end, mark the nodes and handle the details accordingly.
void RooFitDriver::markGPUNodes()
{
   using namespace std::chrono;

   if (_getValInvocations == 1) {
      // leave everything to be computed (and timed) in cpu
      return;
   } else if (_getValInvocations == 2) {
      // compute (and time) as much as possible in gpu
      for (auto &item : _nodeInfos) {
         item.second.computeInGPU = !item.second.computeInScalarMode && item.second.absArg->canComputeBatchWithCuda();
      }
   } else {
      // Assign nodes to gpu using a greedy algorithm: for the number of bytes
      // in this benchmark we take the maximum size of spans in the dataset.
      std::size_t nBytes = 1;
      for (auto const &item : _dataMapCUDA) {
         nBytes = std::max(nBytes, item.second.size() * sizeof(double));
      }
      auto transferTimes = RooFit::CUDAHelpers::memcpyBenchmark(nBytes);

      microseconds h2dTime = transferTimes.first;
      microseconds d2hTime = transferTimes.second;
      ooccoutD(static_cast<RooAbsArg *>(nullptr), FastEvaluations) << "------Copying times------\n";
      ooccoutD(static_cast<RooAbsArg *>(nullptr), FastEvaluations)
         << "h2dTime=" << h2dTime.count() << "us\td2hTime=" << d2hTime.count() << "us\n";

      std::vector<microseconds> diffTimes;
      for (auto &item : _nodeInfos)
         if (!item.second.computeInScalarMode)
            diffTimes.push_back(item.second.cpuTime - item.second.cudaTime);
      microseconds bestTime = microseconds::max(), bestThreshold, ret;
      for (auto &threshold : diffTimes)
         if ((ret = simulateFit(h2dTime, d2hTime, microseconds{std::abs(threshold.count())})) < bestTime) {
            bestTime = ret;
            bestThreshold = threshold;
         }
      // finalize the marking of the best configuration
      simulateFit(h2dTime, d2hTime, microseconds{std::abs(bestThreshold.count())});
      ooccoutD(static_cast<RooAbsArg *>(nullptr), FastEvaluations)
         << "Best threshold=" << bestThreshold.count() << "us" << std::endl;

      // deletion of the timing events (to be replaced later by non-timing events)
      for (auto &item : _nodeInfos) {
         item.second.copyAfterEvaluation = false;
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(item.second.event);
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(item.second.eventStart);
         item.second.event = item.second.eventStart = nullptr;
      }
   } // else (_getValInvocations > 2)

   for (auto &item : _nodeInfos) {
      // scalar nodes don't need copying
      if (!item.second.computeInScalarMode) {
         for (auto *client : item.second.absArg->clients()) {
            if (_nodeInfos.count(client) == 0)
               continue;
            auto &info = _nodeInfos.at(client);
            if (item.second.computeInGPU != info.computeInGPU) {
               item.second.copyAfterEvaluation = true;
               break;
            }
         }
      }
   }

   // restore a cudaEventDisableTiming event when necessary
   if (_getValInvocations == 3) {
      for (auto &item : _nodeInfos)
         if (item.second.computeInGPU || item.second.copyAfterEvaluation)
            item.second.event = RooBatchCompute::dispatchCUDA->newCudaEvent(false);

      ooccoutD(static_cast<RooAbsArg *>(nullptr), FastEvaluations)
         << "------Nodes------\t\t\t\tCpu time: \t Cuda time\n";
      for (auto &item : _nodeInfos)
         ooccoutD(static_cast<RooAbsArg *>(nullptr), FastEvaluations)
            << std::setw(20) << item.first->GetName() << "\t" << item.second.absArg << "\t"
            << (item.second.computeInGPU ? "CUDA" : "CPU") << "\t" << item.second.cpuTime.count() << "us\t"
            << item.second.cudaTime.count() << "us\n";
   }
}

void RooFitDriver::determineOutputSizes()
{
   for (auto *arg : _orderedNodes) {
      auto &argInfo = _nodeInfos.at(arg);
      for (auto *server : arg->servers()) {
         if (server->isValueServer(*arg)) {
            if (!arg->isReducerNode()) {
               argInfo.outputSize = std::max(_nodeInfos.at(server).outputSize, argInfo.outputSize);
            }
         }
      }
   }
}

bool RooFitDriver::isInComputationGraph(RooAbsArg const *arg) const
{
   auto found = _nodeInfos.find(arg);
   return found != _nodeInfos.end() && found->second.absArg == arg;
}

/// Temporarily change the operation mode of a RooAbsArg until the
/// RooFitDriver gets deleted.
void RooFitDriver::setOperMode(RooAbsArg *arg, RooAbsArg::OperMode opMode)
{
   if (opMode != arg->operMode()) {
      _changeOperModeRAIIs.emplace(arg, opMode);
   }
}

} // namespace Experimental
} // namespace ROOT
