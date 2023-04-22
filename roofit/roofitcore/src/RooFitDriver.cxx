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

#include "RooFitDriver.h"

#include <RooAbsCategory.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooRealVar.h>
#include <RooBatchCompute.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooBatchCompute/Initialisation.h>
#include "RooFit/BatchModeDataHelpers.h"
#include "RooFit/BatchModeHelpers.h"
#include "RooFit/CUDAHelpers.h"
#include <RooSimultaneous.h>

#include <TList.h>

#include <iomanip>
#include <numeric>
#include <thread>

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

   bool isScalar() const { return outputSize == 1; }

   bool computeInGPU() const { return !isScalar() && absArg->canComputeBatchWithCuda(); }

   RooAbsArg *absArg = nullptr;

   Detail::AbsBuffer *buffer = nullptr;
   std::size_t iNode = 0;
   cudaEvent_t *event = nullptr;
   cudaStream_t *stream = nullptr;
   int remClients = 0;
   int remServers = 0;
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
/// \param[in] batchMode The computation mode, accepted values are
///            `RooBatchCompute::Cpu` and `RooBatchCompute::Cuda`.
RooFitDriver::RooFitDriver(const RooAbsReal &absReal, RooFit::BatchModeOption batchMode)
   : _topNode{const_cast<RooAbsReal &>(absReal)}, _batchMode{batchMode}
{
   // Initialize RooBatchCompute
   RooBatchCompute::init();

   // Some checks and logging of used architectures
   RooFit::BatchModeHelpers::logArchitectureInfo(_batchMode);

   RooArgSet serverSet;
   RooHelpers::getSortedComputationGraph(topNode(), serverSet);

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
         info.stream = RooBatchCompute::dispatchCUDA->newCudaStream();
      }
   }
}

void RooFitDriver::setData(RooAbsData const &data, std::string const &rangeName, RooSimultaneous const *simPdf,
                           bool skipZeroWeights, bool takeGlobalObservablesFromData)
{
   std::vector<std::pair<std::string, RooAbsData const *>> datas;
   std::vector<bool> isBinnedL;
   bool splitRange = false;

   if (simPdf) {
      _splittedDataSets.clear();
      std::unique_ptr<TList> splits{data.split(*simPdf, true)};
      for (auto *d : static_range_cast<RooAbsData *>(*splits)) {
         RooAbsPdf *simComponent = simPdf->getPdf(d->GetName());
         // If there is no PDF for that component, we also don't need to fill the data
         if (!simComponent) {
            continue;
         }
         datas.emplace_back(std::string("_") + d->GetName() + "_", d);
         isBinnedL.emplace_back(simComponent->getAttribute("BinnedLikelihoodActive"));
         // The dataset need to be kept alive because the datamap points to their content
         _splittedDataSets.emplace_back(d);
      }
      splitRange = simPdf->getAttribute("SplitRange");
   } else {
      datas.emplace_back("", &data);
      isBinnedL.emplace_back(false);
   }

   DataSpansMap dataSpans;

   std::stack<std::vector<double>>{}.swap(_vectorBuffers);

   for (std::size_t iData = 0; iData < datas.size(); ++iData) {
      auto const &toAdd = datas[iData];
      DataSpansMap spans = RooFit::BatchModeDataHelpers::getDataSpans(
         *toAdd.second, RooHelpers::getRangeNameForSimComponent(rangeName, splitRange, toAdd.second->GetName()),
         toAdd.first, _vectorBuffers, skipZeroWeights && !isBinnedL[iData]);
      for (auto const &item : spans) {
         dataSpans.insert(item);
      }
   }

   if (takeGlobalObservablesFromData && data.getGlobalObservables()) {
      _vectorBuffers.emplace();
      auto &buffer = _vectorBuffers.top();
      buffer.reserve(data.getGlobalObservables()->size());
      for (auto *arg : static_range_cast<RooRealVar const *>(*data.getGlobalObservables())) {
         buffer.push_back(arg->getVal());
         dataSpans[arg] = RooSpan<const double>{&buffer.back(), 1};
      }
   }
   setData(dataSpans);
}

void RooFitDriver::setData(DataSpansMap const &dataSpans)
{
   auto outputSizeMap = RooFit::BatchModeDataHelpers::determineOutputSizes(topNode(), dataSpans);

   // Iterate over the given data spans and add them to the data map. Check if
   // they are used in the computation graph. If yes, add the span to the data
   // map and set the node info accordingly.
   std::size_t totalSize = 0;
   for (auto &info : _nodes) {
      if (info.buffer) {
         delete info.buffer;
         info.buffer = nullptr;
      }
      auto found = dataSpans.find(info.absArg->namePtr());
      if (found != dataSpans.end()) {
         _dataMapCPU.at(info.absArg) = found->second;
         info.fromDataset = true;
         info.isDirty = false;
         totalSize += found->second.size();
      } else {
         info.fromDataset = false;
         info.isDirty = true;
      }
   }

   for (auto &info : _nodes) {
      info.outputSize = outputSizeMap.at(info.absArg);

      // In principle we don't need dirty flag propagation because the driver
      // takes care of deciding which node needs to be re-evaluated. However,
      // disabling it also for scalar mode results in very long fitting times
      // for specific models (test 14 in stressRooFit), which still needs to be
      // understood. TODO.
      if (!info.isScalar()) {
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

   markGPUNodes();
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

   double *buffer = nullptr;
   if (nOut == 1) {
      if (_batchMode == RooFit::BatchModeOption::Cuda) {
         _dataMapCUDA.at(node) = RooSpan<const double>(&info.scalarBuffer, nOut);
      }
      buffer = &info.scalarBuffer;
   } else {
      if (!info.buffer) {
         info.buffer = info.copyAfterEvaluation ? _bufferManager.makePinnedBuffer(nOut, info.stream)
                                                : _bufferManager.makeCpuBuffer(nOut);
      }
      buffer = info.buffer->cpuWritePtr();
   }
   _dataMapCPU.at(node) = RooSpan<const double>(buffer, nOut);
   nodeAbsReal->computeBatch(nullptr, buffer, nOut, _dataMapCPU);
   if (info.copyAfterEvaluation) {
      _dataMapCUDA.at(node) = RooSpan<const double>(info.buffer->gpuReadPtr(), nOut);
      if (info.event) {
         RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
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

   // find initial GPU nodes and assign them to GPU
   for (auto &info : _nodes) {
      if (info.remServers == 0 && info.computeInGPU()) {
         assignToGPU(info);
      }
   }

   NodeInfo const &topNodeInfo = _nodes.back();
   while (topNodeInfo.remServers != -2) {
      // find finished GPU nodes
      for (auto &info : _nodes) {
         if (info.remServers == -1 && !RooBatchCompute::dispatchCUDA->streamIsActive(info.stream)) {
            info.remServers = -2;
            // Decrement number of remaining servers for clients and start GPU computations
            for (auto *infoClient : info.clientInfos) {
               --infoClient->remServers;
               if (infoClient->computeInGPU() && infoClient->remServers == 0) {
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
         if (it->remServers == 0 && !it->computeInGPU())
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
         if (--infoClient->remServers == 0 && infoClient->computeInGPU()) {
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
   node->computeBatch(info.stream, buffer, nOut, _dataMapCUDA);
   RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
   if (info.copyAfterEvaluation) {
      _dataMapCPU.at(node) = RooSpan<const double>(info.buffer->cpuReadPtr(), nOut);
   }
}

/// Decides which nodes are assigned to the GPU in a CUDA fit.
void RooFitDriver::markGPUNodes()
{
   for (auto &info : _nodes) {
      info.copyAfterEvaluation = false;
      // scalar nodes don't need copying
      if (!info.isScalar()) {
         for (auto *clientInfo : info.clientInfos) {
            if (info.computeInGPU() != clientInfo->computeInGPU()) {
               info.copyAfterEvaluation = true;
               break;
            }
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
   return _topNode;
}

void RooFitDriver::print(std::ostream &os) const
{
   std::cout << "--- RooFit BatchMode evaluation ---\n";

   std::vector<int> widths{9, 37, 20, 9, 10, 20};

   auto printElement = [&](int iCol, auto const &t) {
      const char separator = ' ';
      os << separator << std::left << std::setw(widths[iCol]) << std::setfill(separator) << t;
      os << "|";
   };

   auto printHorizontalRow = [&]() {
      int n = 0;
      for (int w : widths) {
         n += w + 2;
      }
      for (int i = 0; i < n; i++) {
         os << '-';
      }
      os << "|\n";
   };

   printHorizontalRow();

   os << "|";
   printElement(0, "Index");
   printElement(1, "Name");
   printElement(2, "Class");
   printElement(3, "Size");
   printElement(4, "From Data");
   printElement(5, "1st value");
   std::cout << "\n";

   printHorizontalRow();

   for (std::size_t iNode = 0; iNode < _nodes.size(); ++iNode) {
      auto &nodeInfo = _nodes[iNode];
      RooAbsArg *node = nodeInfo.absArg;

      auto span = _dataMapCPU.at(node);

      os << "|";
      printElement(0, iNode);
      printElement(1, node->GetName());
      printElement(2, node->ClassName());
      printElement(3, nodeInfo.outputSize);
      printElement(4, nodeInfo.fromDataset);
      printElement(5, span[0]);

      std::cout << "\n";
   }

   printHorizontalRow();
}

} // namespace Experimental
} // namespace ROOT
