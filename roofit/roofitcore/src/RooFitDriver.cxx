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

namespace {

void logArchitectureInfo(RooFit::BatchModeOption batchMode)
{
   // We have to exit early if the message stream is not active. Otherwise it's
   // possible that this function skips logging because it thinks it has
   // already logged, but actually it didn't.
   if (!RooMsgService::instance().isActive(static_cast<RooAbsArg *>(nullptr), RooFit::Fitting, RooFit::INFO)) {
      return;
   }

   // Don't repeat logging architecture info if the batchMode option didn't change
   {
      // Second element of pair tracks whether this function has already been called
      static std::pair<RooFit::BatchModeOption, bool> lastBatchMode;
      if (lastBatchMode.second && lastBatchMode.first == batchMode)
         return;
      lastBatchMode = {batchMode, true};
   }

   auto log = [](std::string_view message) {
      oocxcoutI(static_cast<RooAbsArg *>(nullptr), Fitting) << message << std::endl;
   };

   if (batchMode == RooFit::BatchModeOption::Cuda && !RooBatchCompute::dispatchCUDA) {
      throw std::runtime_error(std::string("In: ") + __func__ + "(), " + __FILE__ + ":" + __LINE__ +
                               ": Cuda implementation of the computing library is not available\n");
   }
   if (RooBatchCompute::dispatchCPU->architecture() == RooBatchCompute::Architecture::GENERIC) {
      log("using generic CPU library compiled with no vectorizations");
   } else {
      log(std::string("using CPU computation library compiled with -m") +
          RooBatchCompute::dispatchCPU->architectureName());
   }
   if (batchMode == RooFit::BatchModeOption::Cuda) {
      log("using CUDA computation library");
   }
}

} // namespace

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

   bool computeInGPU() const { return (absArg->isReducerNode() || !isScalar()) && absArg->canComputeBatchWithCuda(); }

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
   bool hasLogged = false;
   std::size_t outputSize = 1;
   std::size_t lastSetValCount = std::numeric_limits<std::size_t>::max();
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
   logArchitectureInfo(_batchMode);

   RooArgSet serverSet;
   RooHelpers::getSortedComputationGraph(topNode(), serverSet);

   _dataMapCPU.resize(serverSet.size());
   _dataMapCUDA.resize(serverSet.size());

   std::map<RooFit::Detail::DataKey, NodeInfo *> nodeInfos;

   // Fill the ordered nodes list and initialize the node info structs.
   _nodes.resize(serverSet.size());
   std::size_t iNode = 0;
   for (RooAbsArg *arg : serverSet) {

      auto &nodeInfo = _nodes[iNode];
      nodeInfo.absArg = arg;
      nodeInfo.iNode = iNode;
      nodeInfos[arg] = &nodeInfo;

      if (dynamic_cast<RooRealVar const *>(arg)) {
         nodeInfo.isVariable = true;
      } else {
         arg->setDataToken(iNode);
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
      }
   }

   syncDataTokens();

   if (_batchMode == RooFit::BatchModeOption::Cuda) {
      // create events and streams for every node
      for (auto &info : _nodes) {
         info.event = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
         info.stream = RooBatchCompute::dispatchCUDA->newCudaStream();
      }
   }
}

/// If there are servers with the same name that got de-duplicated in the
/// `_nodes` list, we need to set their data tokens too. We find such nodes by
/// visiting the servers of every known node.
void RooFitDriver::syncDataTokens()
{
   for (NodeInfo &info : _nodes) {
      std::size_t iValueServer = 0;
      for (RooAbsArg *server : info.absArg->servers()) {
         if (server->isValueServer(*info.absArg)) {
            auto *knownServer = info.serverInfos[iValueServer]->absArg;
            if (knownServer->hasDataToken()) {
               server->setDataToken(knownServer->dataToken());
            }
            ++iValueServer;
         }
      }
   }
}

void RooFitDriver::setData(RooAbsData const &data, std::string const &rangeName, RooSimultaneous const *simPdf,
                           bool skipZeroWeights, bool takeGlobalObservablesFromData)
{
   std::stack<std::vector<double>>{}.swap(_vectorBuffers);
   setData(RooFit::BatchModeDataHelpers::getDataSpans(data, rangeName, simPdf, skipZeroWeights,
                                                      takeGlobalObservablesFromData, _vectorBuffers));
}

void RooFitDriver::setData(DataSpansMap const &dataSpans)
{
   auto outputSizeMap = RooFit::BatchModeDataHelpers::determineOutputSizes(topNode(), dataSpans);

   // Iterate over the given data spans and add them to the data map. Check if
   // they are used in the computation graph. If yes, add the span to the data
   // map and set the node info accordingly.
   std::size_t totalSize = 0;
   std::size_t iNode = 0;
   for (auto &info : _nodes) {
      if (info.buffer) {
         delete info.buffer;
         info.buffer = nullptr;
      }
      auto found = dataSpans.find(info.absArg->namePtr());
      if (found != dataSpans.end()) {
         info.absArg->setDataToken(iNode);
         _dataMapCPU.set(info.absArg, found->second);
         info.fromDataset = true;
         info.isDirty = false;
         totalSize += found->second.size();
      } else {
         info.fromDataset = false;
         info.isDirty = true;
      }
      ++iNode;
   }

   syncDataTokens();

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
      if (size == 1) {
         // Scalar observables from the data don't need to be copied to the GPU
         _dataMapCUDA.set(info.absArg, _dataMapCPU.at(info.absArg));
      } else {
         _dataMapCUDA.set(info.absArg, {_cudaMemDataset + idx, size});
         RooBatchCompute::dispatchCUDA->memcpyToCUDA(_cudaMemDataset + idx, _dataMapCPU.at(info.absArg).data(),
                                                     size * sizeof(double));
         idx += size;
      }
   }

   markGPUNodes();
}

RooFitDriver::~RooFitDriver()
{
   for (auto &info : _nodes) {
      info.absArg->resetDataToken();
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
      buffer = &info.scalarBuffer;
      if (_batchMode == RooFit::BatchModeOption::Cuda) {
         _dataMapCUDA.set(node, {buffer, nOut});
      }
   } else {
      if (!info.hasLogged && _batchMode == RooFit::BatchModeOption::Cuda) {
         RooAbsArg const &arg = *info.absArg;
         oocoutI(&arg, FastEvaluations) << "The argument " << arg.ClassName() << "::" << arg.GetName()
                                        << " could not be evaluated on the GPU because the class doesn't support it. "
                                           "Consider requesting or implementing it to benefit from a speed up."
                                        << std::endl;
         info.hasLogged = true;
      }
      if (!info.buffer) {
         info.buffer = info.copyAfterEvaluation ? _bufferManager.makePinnedBuffer(nOut, info.stream)
                                                : _bufferManager.makeCpuBuffer(nOut);
      }
      buffer = info.buffer->cpuWritePtr();
   }
   _dataMapCPU.set(node, {buffer, nOut});
   nodeAbsReal->computeBatch(nullptr, buffer, nOut, _dataMapCPU);
   if (info.copyAfterEvaluation) {
      _dataMapCUDA.set(node, {info.buffer->gpuReadPtr(), nOut});
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

   info.remServers = -1;
   // wait for every server to finish
   for (auto *infoServer : info.serverInfos) {
      if (infoServer->event)
         RooBatchCompute::dispatchCUDA->cudaStreamWaitEvent(info.stream, infoServer->event);
   }

   const std::size_t nOut = info.outputSize;

   double *buffer = nullptr;
   if (nOut == 1) {
      buffer = &info.scalarBuffer;
      _dataMapCPU.set(node, {buffer, nOut});
   } else {
      info.buffer = info.copyAfterEvaluation ? _bufferManager.makePinnedBuffer(nOut, info.stream)
                                             : _bufferManager.makeGpuBuffer(nOut);
      buffer = info.buffer->gpuWritePtr();
   }
   _dataMapCUDA.set(node, {buffer, nOut});
   node->computeBatch(info.stream, buffer, nOut, _dataMapCUDA);
   RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
   if (info.copyAfterEvaluation) {
      _dataMapCPU.set(node, {info.buffer->cpuReadPtr(), nOut});
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

RooAbsRealWrapper::RooAbsRealWrapper(std::unique_ptr<RooFitDriver> driver, std::string const &rangeName,
                                     RooSimultaneous const *simPdf, bool takeGlobalObservablesFromData)
   : RooAbsReal{"RooFitDriverWrapper", "RooFitDriverWrapper"},
     _driver{std::move(driver)},
     _topNode("topNode", "top node", this, _driver->topNode()),
     _rangeName{rangeName},
     _simPdf{simPdf},
     _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
{
}

RooAbsRealWrapper::RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name)
   : RooAbsReal{other, name},
     _driver{other._driver},
     _topNode("topNode", this, other._topNode),
     _data{other._data},
     _parameters{other._parameters},
     _rangeName{other._rangeName},
     _simPdf{other._simPdf},
     _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData}
{
}

bool RooAbsRealWrapper::getParameters(const RooArgSet *observables, RooArgSet &outputSet,
                                      bool /*stripDisconnected*/) const
{
   outputSet.add(_parameters);
   if (observables) {
      outputSet.remove(*observables);
   }
   // If we take the global observables as data, we have to return these as
   // parameters instead of the parameters in the model. Otherwise, the
   // constant parameters in the fit result that are global observables will
   // not have the right values.
   if (_takeGlobalObservablesFromData && _data->getGlobalObservables()) {
      outputSet.replace(*_data->getGlobalObservables());
   }
   return false;
}

bool RooAbsRealWrapper::setData(RooAbsData &data, bool /*cloneData*/)
{
   _data = &data;

   // Figure out what are the parameters for the current dataset
   _parameters.clear();
   RooArgSet params;
   _driver->topNode().getParameters(_data->get(), params, true);
   for (RooAbsArg *param : params) {
      if (!param->getAttribute("__obs__")) {
         _parameters.add(*param);
      }
   }

   _driver->setData(*_data, _rangeName, _simPdf, /*skipZeroWeights=*/true, _takeGlobalObservablesFromData);
   return true;
}

} // namespace Experimental
} // namespace ROOT
