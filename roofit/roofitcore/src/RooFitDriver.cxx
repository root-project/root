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
#include <RooRealVar.h>
#include <RooArgList.h>
#include <RooBatchCompute.h>
#include <RooMsgService.h>
#include <RooBatchCompute/Initialisation.h>
#include <RooFit/BatchModeDataHelpers.h>
#include <RooFit/BatchModeHelpers.h>
#include <RooFit/CUDAHelpers.h>

#include "NormalizationHelpers.h"

#include <iomanip>
#include <numeric>
#include <thread>

#define COUT_DEBUG ooccoutD(nullptr, FastEvaluations)

namespace {

enum HeterogeneosIterations { CPUOnly = 2, GPUOnly = 1, Both = 3 };

}

namespace ROOT {
namespace Experimental {

/// A struct used by the RooFitDriver to store information on the RooAbsArgs in
/// the computation graph.
struct NodeInfo {
   /// Check the servers of a node that has been computed and release it's resources
   /// if they are no longer needed.
   void decrementRemainingClients()
   {
      if (--remClients == 0) {
         delete buffer;
         buffer = nullptr;
      }
   }

   RooAbsArg *absArg = nullptr;

   Detail::AbsBuffer *buffer = nullptr;
   std::size_t iNode = 0;
   cudaEvent_t *event = nullptr;
   cudaEvent_t *eventStart = nullptr;
   cudaStream_t *stream = nullptr;
   std::chrono::microseconds cpuTime{0};
   std::chrono::microseconds cudaTime{std::chrono::microseconds::max()};
   int remClients = 0;
   int remServers = 0;
   bool isScalar = false;
   bool computeInGPU = false;
   bool copyAfterEvaluation = false;
   bool fromDataset = false;
   bool isVariable = false;
   bool isDirty = true;
   bool isCategory = false;
   std::size_t outputSize = 1;
   std::size_t lastSetValCount = std::numeric_limits<std::size_t>::max();
   std::size_t originalDataToken = 0;
   double scalarBuffer = 0.0;
   std::vector<NodeInfo *> serverInfos;
   std::vector<NodeInfo *> clientInfos;

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
/// \param[in] absReal The RooAbsReal object that sits on top of the
///            computation graph that we want to evaluate.
/// \param[in] normSet Normalization set for the evaluation
/// \param[in] batchMode The computation mode, accepted values are
///            `RooBatchCompute::Cpu` and `RooBatchCompute::Cuda`.
RooFitDriver::RooFitDriver(const RooAbsReal &absReal, RooArgSet const &normSet, RooFit::BatchModeOption batchMode)
   : _batchMode{batchMode}
{
   _integralUnfolder = std::make_unique<RooFit::NormalizationIntegralUnfolder>(absReal, normSet);

   // Initialize RooBatchCompute
   RooBatchCompute::init();

   // Some checks and logging of used architectures
   RooFit::BatchModeHelpers::logArchitectureInfo(_batchMode);

   // Get the set of nodes in the computation graph. Do the detour via
   // RooArgList to avoid deduplication done after adding each element.
   RooArgList serverList;
   topNode().treeNodeServerList(&serverList, nullptr, true, true, false, true);
   // If we fill the servers in reverse order, they are approximately in
   // topological order so we save a bit of work in sortTopologically().
   RooArgSet serverSet;
   serverSet.add(serverList.rbegin(), serverList.rend(), /*silent=*/true);
   // Sort nodes topologically: the servers of any node will be before that
   // node in the collection.
   serverSet.sortTopologically();

   _dataMapCPU.resize(serverSet.size());
   _dataMapCUDA.resize(serverSet.size());

   std::unordered_map<TNamed const *, std::size_t> tokens;
   std::map<RooFit::Detail::DataKey, NodeInfo *> nodeInfos;

   // Fill the ordered nodes list and initialize the node info structs.
   _nodes.resize(serverSet.size());
   std::size_t iNode = 0;
   for (RooAbsArg *arg : serverSet) {

      tokens[arg->namePtr()] = iNode;

      auto &nodeInfo = _nodes[iNode];
      nodeInfo.absArg = arg;
      nodeInfo.iNode = iNode;
      nodeInfos[arg] = &nodeInfo;

      nodeInfo.originalDataToken = arg->dataToken();
      arg->setDataToken(iNode);

      if (dynamic_cast<RooRealVar const *>(arg)) {
         nodeInfo.isVariable = true;
      }
      if (dynamic_cast<RooAbsCategory const *>(arg)) {
         nodeInfo.isCategory = true;
      }

      ++iNode;
   }

   for (NodeInfo &info : _nodes) {
      info.serverInfos.reserve(info.absArg->servers().size());
      for (RooAbsArg *server : info.absArg->servers()) {
         if (server->isValueServer(*info.absArg)) {
            auto *serverInfo = nodeInfos.at(server);
            info.serverInfos.emplace_back(serverInfo);
            serverInfo->clientInfos.emplace_back(&info);
         }
         server->setDataToken(tokens.at(server->namePtr()));
      }
   }

   if (_batchMode == RooFit::BatchModeOption::Cuda) {
      // create events and streams for every node
      for (auto &info : _nodes) {
         info.event = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
         info.eventStart = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
         info.stream = RooBatchCompute::dispatchCUDA->newCudaStream();
      }
   }
}

void RooFitDriver::setData(RooAbsData const &data, std::string_view rangeName,
                           RooAbsCategory const *indexCatForSplitting, bool skipZeroWeights)
{
   setData(RooFit::BatchModeDataHelpers::getDataSpans(data, rangeName, indexCatForSplitting, _vectorBuffers,
                                                      skipZeroWeights));
}

void RooFitDriver::setData(DataSpansMap const &dataSpans)
{
   // Iterate over the given data spans and add them to the data map. Check if
   // they are used in the computation graph. If yes, add the span to the data
   // map and set the node info accordingly.
   std::size_t totalSize = 0;
   for (auto &info : _nodes) {
      auto found = dataSpans.find(info.absArg->namePtr());
      if (found != dataSpans.end()) {
         _dataMapCPU.at(info.absArg) = found->second;
         info.outputSize = found->second.size();
         info.fromDataset = true;
         info.isDirty = false;
         totalSize += info.outputSize;
      }
   }

   determineOutputSizes();

   for (auto &info : _nodes) {
      // If the node has an output of size 1
      info.isScalar = info.outputSize == 1;

      // In principle we don't need dirty flag propagation because the driver
      // takes care of deciding which node needs to be re-evaluated. However,
      // disabling it also for scalar mode results in very long fitting times
      // for specific models (test 14 in stressRooFit), which still needs to be
      // understood. TODO.
      if (!info.isScalar) {
         setOperMode(info.absArg, RooAbsArg::ADirty);
      }
   }

   // Extra steps for initializing in cuda mode
   if (_batchMode != RooFit::BatchModeOption::Cuda)
      return;

   // copy observable data to the GPU
   // TODO: use separate buffers here
   _cudaMemDataset = static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMalloc(totalSize * sizeof(double)));
   size_t idx = 0;
   for (auto &info : _nodes) {
      if (!info.fromDataset)
         continue;
      std::size_t size = info.outputSize;
      _dataMapCUDA.at(info.absArg) = RooSpan<double>(_cudaMemDataset + idx, size);
      RooBatchCompute::dispatchCUDA->memcpyToCUDA(_cudaMemDataset + idx, _dataMapCPU.at(info.absArg).data(),
                                                  size * sizeof(double));
      idx += size;
   }
}

RooFitDriver::~RooFitDriver()
{
   for (auto &info : _nodes) {
      info.absArg->setDataToken(info.originalDataToken);
   }

   if (_batchMode == RooFit::BatchModeOption::Cuda) {
      RooBatchCompute::dispatchCUDA->cudaFree(_cudaMemDataset);
   }
}

std::vector<double> RooFitDriver::getValues()
{
   getVal();
   NodeInfo const &nodeInfo = _nodes.back();
   if (nodeInfo.computeInGPU) {
      std::size_t nOut = nodeInfo.outputSize;
      std::vector<double> out(nOut);
      RooBatchCompute::dispatchCUDA->memcpyToCPU(out.data(), _dataMapCPU.at(&topNode()).data(), nOut * sizeof(double));
      _dataMapCPU.at(&topNode()) = RooSpan<const double>(out.data(), nOut);
      return out;
   }
   // We copy the data to the output vector
   auto dataSpan = _dataMapCPU.at(&topNode());
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

   auto nodeAbsReal = static_cast<RooAbsReal const *>(node);

   const std::size_t nOut = info.outputSize;

   if (nOut == 1) {
      _dataMapCPU.at(node) = RooSpan<const double>(&info.scalarBuffer, nOut);
      if (_batchMode == RooFit::BatchModeOption::Cuda) {
         _dataMapCUDA.at(node) = RooSpan<const double>(&info.scalarBuffer, nOut);
      }
      nodeAbsReal->computeBatch(nullptr, &info.scalarBuffer, nOut, _dataMapCPU);
   } else {
      if (!info.buffer) {
         info.buffer = info.copyAfterEvaluation ? _bufferManager.makePinnedBuffer(nOut, info.stream)
                                                : _bufferManager.makeCpuBuffer(nOut);
      }
      double *buffer = info.buffer->cpuWritePtr();
      _dataMapCPU.at(node) = RooSpan<const double>(buffer, nOut);
      // compute node and measure the time the first time
      if (_getValInvocations == CPUOnly) {
         using namespace std::chrono;
         auto start = steady_clock::now();
         nodeAbsReal->computeBatch(nullptr, buffer, nOut, _dataMapCPU);
         info.cpuTime = duration_cast<microseconds>(steady_clock::now() - start);
      } else {
         nodeAbsReal->computeBatch(nullptr, buffer, nOut, _dataMapCPU);
      }
      if (info.copyAfterEvaluation) {
         _dataMapCUDA.at(node) = RooSpan<const double>(info.buffer->gpuReadPtr(), nOut);
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

   if (_batchMode == RooFit::BatchModeOption::Cuda) {
      return getValHeterogeneous();
   }

   for (auto &nodeInfo : _nodes) {
      RooAbsArg *node = nodeInfo.absArg;
      if (!nodeInfo.fromDataset) {
         if (nodeInfo.isVariable) {
            auto *var = static_cast<RooRealVar const *>(node);
            if (nodeInfo.lastSetValCount != var->valueResetCounter()) {
               nodeInfo.lastSetValCount = var->valueResetCounter();
               for (NodeInfo *clientInfo : nodeInfo.clientInfos) {
                  clientInfo->isDirty = true;
               }
               computeCPUNode(node, nodeInfo);
               nodeInfo.isDirty = false;
            }
         } else {
            if (nodeInfo.isDirty) {
               for (NodeInfo *clientInfo : nodeInfo.clientInfos) {
                  clientInfo->isDirty = true;
               }
               computeCPUNode(node, nodeInfo);
               nodeInfo.isDirty = false;
            }
         }
      }
   }

   // return the final value
   return _dataMapCPU.at(&topNode())[0];
}

/// Returns the value of the top node in the computation graph
double RooFitDriver::getValHeterogeneous()
{
   for (auto &info : _nodes) {
      info.remClients = info.clientInfos.size();
      info.remServers = info.serverInfos.size();
      if (info.buffer)
         delete info.buffer;
      info.buffer = nullptr;
   }

   // In a cuda fit, use first 3 fits to determine the execution times
   // and the hardware that computes each part of the graph
   if (_getValInvocations <= Both) {
      // leave everything to be computed (and timed) in CPU in the 1st
      // invocation, and after the 3rd the GPU nodes are already marked.
      markGPUNodes();
   }

   // find initial GPU nodes and assign them to GPU
   for (auto &info : _nodes) {
      if (info.remServers == 0 && info.computeInGPU) {
         assignToGPU(info);
      }
   }

   NodeInfo const &topNodeInfo = _nodes.back();
   while (topNodeInfo.remServers != -2) {
      // find finished GPU nodes
      for (auto &info : _nodes) {
         if (info.remServers == -1 && !RooBatchCompute::dispatchCUDA->streamIsActive(info.stream)) {
            if (_getValInvocations == GPUOnly) {
               float ms = RooBatchCompute::dispatchCUDA->cudaEventElapsedTime(info.eventStart, info.event);
               info.cudaTime += std::chrono::microseconds{int(1000.0 * ms)};
            }
            info.remServers = -2;
            // Decrement number of remaining servers for clients and start GPU computations
            for (auto *infoClient : info.clientInfos) {
               --infoClient->remServers;
               if (infoClient->computeInGPU && infoClient->remServers == 0) {
                  assignToGPU(*infoClient);
               }
            }
            for (auto *serverInfo : info.serverInfos) {
               serverInfo->decrementRemainingClients();
            }
         }
      }

      // find next CPU node
      auto it = _nodes.begin();
      for (; it != _nodes.end(); it++) {
         if (it->remServers == 0 && !it->computeInGPU)
            break;
      }

      // if no CPU node available sleep for a while to save CPU usage
      if (it == _nodes.end()) {
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
         continue;
      }

      // compute next CPU node
      NodeInfo &info = *it;
      RooAbsArg const *node = info.absArg;
      info.remServers = -2; // so that it doesn't get picked again

      if (!info.fromDataset) {
         computeCPUNode(node, info);
      }

      // Assign the clients that are computed on the GPU
      for (auto *infoClient : info.clientInfos) {
         if (--infoClient->remServers == 0 && infoClient->computeInGPU) {
            assignToGPU(*infoClient);
         }
      }
      for (auto *serverInfo : info.serverInfos) {
         serverInfo->decrementRemainingClients();
      }
   }

   // return the final value
   return _dataMapCPU.at(&topNode())[0];
}

/// Assign a node to be computed in the GPU. Scan it's clients and also assign them
/// in case they only depend on GPU nodes.
void RooFitDriver::assignToGPU(NodeInfo &info)
{
   using namespace Detail;

   auto node = static_cast<RooAbsReal const *>(info.absArg);

   const std::size_t nOut = info.outputSize;

   info.remServers = -1;
   // wait for every server to finish
   for (auto *infoServer : info.serverInfos) {
      if (infoServer->event)
         RooBatchCompute::dispatchCUDA->cudaStreamWaitEvent(info.stream, infoServer->event);
   }

   info.buffer = info.copyAfterEvaluation ? _bufferManager.makePinnedBuffer(nOut, info.stream)
                                          : _bufferManager.makeGpuBuffer(nOut);
   double *buffer = info.buffer->gpuWritePtr();
   _dataMapCUDA.at(node) = RooSpan<const double>(buffer, nOut);
   // measure launching overhead (add computation time later)
   if (_getValInvocations == GPUOnly) {
      using namespace std::chrono;
      RooBatchCompute::dispatchCUDA->cudaEventRecord(info.eventStart, info.stream);
      auto start = steady_clock::now();
      node->computeBatch(info.stream, buffer, nOut, _dataMapCUDA);
      info.cudaTime = duration_cast<microseconds>(steady_clock::now() - start);
   } else {
      node->computeBatch(info.stream, buffer, nOut, _dataMapCUDA);
   }
   RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
   if (info.copyAfterEvaluation) {
      _dataMapCPU.at(node) = RooSpan<const double>(info.buffer->cpuReadPtr(), nOut);
   }
}

} // namespace Experimental
} // namespace ROOT

namespace {

/// This methods simulates the computation of the whole graph and the time it takes
/// and decides what to compute in GPU. The decision is made on the basis of avoiding
/// leaving either the GPU or the CPU idle at any time, if possible, and on assigning
/// to each piece of hardware a computation that is significantly slower on the other part.
/// The nodes may be assigned to the non-efficient side (CPU or GPU) to prevent idleness
/// only if the absolute difference cpuTime-cudaTime does not exceed the diffThreshold.
std::chrono::microseconds simulateFit(std::vector<bool> &computeInGPU,
                                      std::vector<ROOT::Experimental::NodeInfo> const &nodes,
                                      std::chrono::microseconds h2dTime, std::chrono::microseconds d2hTime,
                                      std::chrono::microseconds diffThreshold)
{
   using namespace std::chrono;
   using ROOT::Experimental::NodeInfo;

   for (std::size_t iNode = 0; iNode < nodes.size(); ++iNode) {
      auto const &info = nodes[iNode];
      computeInGPU[iNode] = !info.isScalar && info.absArg->canComputeBatchWithCuda();
   }

   std::vector<std::chrono::microseconds> timeLaunched;
   timeLaunched.reserve(nodes.size());

   std::size_t nNodes = nodes.size();
   // launch scalar nodes (assume they are computed in 0 time)
   for (auto &info : nodes) {
      if (info.isScalar) {
         nNodes--;
         timeLaunched.emplace_back(0);
      } else
         timeLaunched.emplace_back(-1);
   }

   NodeInfo const *cpuNode = nullptr;
   NodeInfo const *cudaNode = nullptr;
   microseconds simulatedTime{0};
   while (nNodes) {
      microseconds minDiff = microseconds::max(), maxDiff = -minDiff; // diff = cpuTime - cudaTime
      NodeInfo const *cpuCandidate = nullptr;
      NodeInfo const *cudaCandidate = nullptr;
      microseconds cpuDelay{};
      microseconds cudaDelay{};
      for (std::size_t iNode = 0; iNode < nodes.size(); ++iNode) {
         auto &info = nodes[iNode];
         if (timeLaunched[iNode] >= microseconds{0}) {
            // already launched
            continue;
         }
         microseconds diff{info.cpuTime - info.cudaTime}, cpuWait{0}, cudaWait{0};

         bool goToNextCandidate = false;

         for (auto *serverInfo : info.serverInfos) {
            if (serverInfo->isScalar)
               continue;

            // dependencies not computed yet
            if (timeLaunched[serverInfo->iNode] < microseconds{0}) {
               goToNextCandidate = true;
               break;
            }
            if (computeInGPU[serverInfo->iNode]) {
               cpuWait =
                  std::max(cpuWait, timeLaunched[serverInfo->iNode] + serverInfo->cudaTime + d2hTime - simulatedTime);
            } else {
               cudaWait =
                  std::max(cudaWait, timeLaunched[serverInfo->iNode] + serverInfo->cpuTime + h2dTime - simulatedTime);
            }
         }

         if (goToNextCandidate) {
            continue;
         }

         diff += cpuWait - cudaWait;
         if (diff < minDiff) {
            minDiff = diff;
            cpuDelay = cpuWait;
            cpuCandidate = &info;
         }
         if (diff > maxDiff && info.absArg->canComputeBatchWithCuda()) {
            maxDiff = diff;
            cudaDelay = cudaWait;
            cudaCandidate = &info;
         }
      }

      auto calcDiff = [](const NodeInfo *nodeInfo) { return nodeInfo->cpuTime - nodeInfo->cudaTime; };
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
         timeLaunched[cpuNode->iNode] = simulatedTime + cpuDelay;
         computeInGPU[cpuNode->iNode] = false;
         nNodes--;
      }
      if (cudaCandidate && !cudaNode) {
         cudaNode = cudaCandidate;
         timeLaunched[cudaNode->iNode] = simulatedTime + cudaDelay;
         computeInGPU[cudaNode->iNode] = true;
         nNodes--;
      }

      microseconds etaCPU{microseconds::max()}, etaCUDA{microseconds::max()};
      if (cpuNode) {
         etaCPU = timeLaunched[cpuNode->iNode] + cpuNode->cpuTime;
      }
      if (cudaNode) {
         etaCUDA = timeLaunched[cudaNode->iNode] + cudaNode->cudaTime;
      }
      simulatedTime = std::min(etaCPU, etaCUDA);
      if (etaCPU < etaCUDA)
         cpuNode = nullptr;
      else
         cudaNode = nullptr;
   } // while(nNodes)
   return simulatedTime;
}

std::vector<bool> selectNodesForGPU(std::vector<ROOT::Experimental::NodeInfo> const &nodes)
{
   using namespace std::chrono;

   // Assign nodes to GPU using a greedy algorithm: for the number of bytes
   // in this benchmark we take the maximum size of spans in the dataset.
   std::size_t nEvents = 1;
   for (auto const &node : nodes) {
      nEvents = std::max(nEvents, node.outputSize);
   }

   auto transferTimes = RooFit::CUDAHelpers::memcpyBenchmark(nEvents * sizeof(double));

   microseconds h2dTime = transferTimes.first;
   microseconds d2hTime = transferTimes.second;
   COUT_DEBUG << "------Copying times------\n";
   COUT_DEBUG << "h2dTime=" << h2dTime.count() << "us\td2hTime=" << d2hTime.count() << "us\n";

   std::vector<microseconds> diffTimes;
   for (auto &info : nodes) {
      if (!info.isScalar)
         diffTimes.push_back(info.cpuTime - info.cudaTime);
   }
   microseconds bestTime = microseconds::max();
   microseconds bestThreshold{};
   microseconds ret;
   std::vector<bool> computeInGPU(nodes.size());
   for (auto &threshold : diffTimes) {
      if ((ret = simulateFit(computeInGPU, nodes, h2dTime, d2hTime, microseconds{std::abs(threshold.count())})) <
          bestTime) {
         bestTime = ret;
         bestThreshold = threshold;
      }
   }
   // finalize the marking of the best configuration
   simulateFit(computeInGPU, nodes, h2dTime, d2hTime, microseconds{std::abs(bestThreshold.count())});

   COUT_DEBUG << "Best threshold = " << bestThreshold.count() << " us" << std::endl;

   return computeInGPU;
}

} // namespace

namespace ROOT {
namespace Experimental {

/// Decides which nodes are assigned to the GPU in a cuda fit. In the 1st iteration,
/// everything is computed in CPU for measuring the CPU time. In the 2nd iteration,
/// everything is computed in GPU (if possible) to measure the GPU time.
/// In the 3rd iteration, simulate the computation of the graph by calling simulateFit
/// with every distinct threshold found as timeDiff within the nodes of the graph and select
/// the best configuration. In the end, mark the nodes and handle the details accordingly.
void RooFitDriver::markGPUNodes()
{
   if (_getValInvocations == CPUOnly) {
      for (auto &info : _nodes) {
         info.copyAfterEvaluation = false;
         info.computeInGPU = false;
      }
   } else if (_getValInvocations == Both) {
      auto computeInGPU = selectNodesForGPU(_nodes);
      for (std::size_t iNode = 0; iNode < _nodes.size(); ++iNode) {
         _nodes[iNode].copyAfterEvaluation = false;
         _nodes[iNode].computeInGPU = computeInGPU[iNode];
      }

      // deletion of the timing events (to be replaced later by non-timing events)
      for (auto &info : _nodes) {
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(info.event);
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(info.eventStart);
         info.event = info.eventStart = nullptr;
      }
   } else {
      // compute (and time) as much as possible in GPU
      for (auto &info : _nodes) {
         info.copyAfterEvaluation = false;
         info.computeInGPU = !info.isScalar && info.absArg->canComputeBatchWithCuda();
      }
   }

   for (auto &info : _nodes) {
      // scalar nodes don't need copying
      if (!info.isScalar) {
         for (auto *clientInfo : info.clientInfos) {
            if (info.computeInGPU != clientInfo->computeInGPU) {
               info.copyAfterEvaluation = true;
               break;
            }
         }
      }
   }

   // restore a cudaEventDisableTiming event when necessary
   if (_getValInvocations == Both) {
      for (auto &info : _nodes) {
         if (info.computeInGPU || info.copyAfterEvaluation)
            info.event = RooBatchCompute::dispatchCUDA->newCudaEvent(false);
      }

      COUT_DEBUG << "------Nodes------\t\t\t\tCpu time: \t Cuda time\n";
      for (auto &info : _nodes) {
         COUT_DEBUG << std::setw(20) << info.absArg->GetName() << "\t" << info.absArg << "\t"
                    << (info.computeInGPU ? "CUDA" : "CPU") << "\t" << info.cpuTime.count() << "us\t"
                    << info.cudaTime.count() << "us\n";
      }
   }
}

void RooFitDriver::determineOutputSizes()
{
   for (auto &argInfo : _nodes) {
      for (auto *serverInfo : argInfo.serverInfos) {
         if (!argInfo.absArg->isReducerNode()) {
            argInfo.outputSize = std::max(serverInfo->outputSize, argInfo.outputSize);
         }
      }
   }
}

/// Temporarily change the operation mode of a RooAbsArg until the
/// RooFitDriver gets deleted.
void RooFitDriver::setOperMode(RooAbsArg *arg, RooAbsArg::OperMode opMode)
{
   if (opMode != arg->operMode()) {
      _changeOperModeRAIIs.emplace(arg, opMode);
   }
}

RooAbsReal &RooFitDriver::topNode() const
{
   return static_cast<RooAbsReal &>(_integralUnfolder->arg());
}

} // namespace Experimental
} // namespace ROOT
