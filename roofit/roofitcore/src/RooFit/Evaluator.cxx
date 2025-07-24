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
\file Evaluator.cxx
\class RooFit::Evaluator
\ingroup Roofitcore

Evaluates a RooAbsReal object in other ways than recursive graph
traversal. Currently, it is being used for evaluating a RooAbsReal object and
supplying the value to the minimizer, during a fit. The class scans the
dependencies and schedules the computations in a secure and efficient way. The
computations take place in the RooBatchCompute library and can be carried off
by either the CPU or a CUDA-supporting GPU. The Evaluator class takes care
of data transfers. An instance of this class is created every time
RooAbsPdf::fitTo() is called and gets destroyed when the fitting ends.
**/

#include <RooFit/Evaluator.h>

#include <RooAbsCategory.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooRealVar.h>
#include <RooBatchCompute.h>
#include <RooMsgService.h>
#include <RooNameReg.h>
#include <RooSimultaneous.h>

#include <RooBatchCompute.h>

#include "BatchModeDataHelpers.h"
#include "RooFitImplHelpers.h"

#include <chrono>
#include <iomanip>
#include <numeric>
#include <thread>

namespace RooFit {

namespace {

// To avoid deleted move assignment.
template <class T>
void assignSpan(std::span<T> &to, std::span<T> const &from)
{
   to = from;
}

void logArchitectureInfo(bool useGPU)
{
   // We have to exit early if the message stream is not active. Otherwise it's
   // possible that this function skips logging because it thinks it has
   // already logged, but actually it didn't.
   if (!RooMsgService::instance().isActive(nullptr, RooFit::Fitting, RooFit::INFO)) {
      return;
   }

   // Don't repeat logging architecture info if the useGPU option didn't change
   {
      // Second element of pair tracks whether this function has already been called
      static std::pair<bool, bool> lastUseGPU;
      if (lastUseGPU.second && lastUseGPU.first == useGPU)
         return;
      lastUseGPU = {useGPU, true};
   }

   auto log = [](std::string_view message) {
      oocxcoutI(static_cast<RooAbsArg *>(nullptr), Fitting) << message << std::endl;
   };

   if (RooBatchCompute::cpuArchitecture() == RooBatchCompute::Architecture::GENERIC) {
      log("using generic CPU library compiled with no vectorizations");
   } else {
      log(std::string("using CPU computation library compiled with -m") + RooBatchCompute::cpuArchitectureName());
   }
   if (useGPU) {
      log("using CUDA computation library");
   }
}

} // namespace

/// A struct used by the Evaluator to store information on the RooAbsArgs in
/// the computation graph.
struct NodeInfo {

   bool isScalar() const { return outputSize == 1; }

   RooAbsArg *absArg = nullptr;
   RooAbsArg::OperMode originalOperMode;

   std::shared_ptr<RooBatchCompute::AbsBuffer> buffer;
   std::size_t iNode = 0;
   int remClients = 0;
   int remServers = 0;
   bool copyAfterEvaluation = false;
   bool fromArrayInput = false;
   bool isVariable = false;
   bool isDirty = true;
   bool isCategory = false;
   bool hasLogged = false;
   bool computeInGPU = false;
   std::size_t outputSize = 1;
   std::size_t lastSetValCount = std::numeric_limits<std::size_t>::max();
   int lastCatVal = std::numeric_limits<int>::max();
   double scalarBuffer = 0.0;
   std::vector<NodeInfo *> serverInfos;
   std::vector<NodeInfo *> clientInfos;

   RooBatchCompute::CudaInterface::CudaEvent *event = nullptr;
   RooBatchCompute::CudaInterface::CudaStream *stream = nullptr;

   /// Check the servers of a node that has been computed and release its
   /// resources if they are no longer needed.
   void decrementRemainingClients()
   {
      if (--remClients == 0 && !fromArrayInput) {
         buffer.reset();
      }
   }

   ~NodeInfo()
   {
      if (event)
         RooBatchCompute::dispatchCUDA->deleteCudaEvent(event);
      if (stream)
         RooBatchCompute::dispatchCUDA->deleteCudaStream(stream);
   }
};

/// Construct a new Evaluator. The constructor analyzes and saves metadata about the graph,
/// useful for the evaluation of it that will be done later. In case the CUDA mode is selected,
/// there's also some CUDA-related initialization.
///
/// \param[in] absReal The RooAbsReal object that sits on top of the
///            computation graph that we want to evaluate.
/// \param[in] useGPU Whether the evaluation should be preferably done on the GPU.
Evaluator::Evaluator(const RooAbsReal &absReal, bool useGPU)
   : _topNode{const_cast<RooAbsReal &>(absReal)}, _useGPU{useGPU}
{
   RooBatchCompute::initCPU();
   if (useGPU && RooBatchCompute::initCUDA() != 0) {
      throw std::runtime_error("Can't create Evaluator in CUDA mode because RooBatchCompute CUDA could not be loaded!");
   }
   // Some checks and logging of used architectures
   logArchitectureInfo(_useGPU);

   _bufferManager = _useGPU ? RooBatchCompute::dispatchCUDA->createBufferManager()
                            : RooBatchCompute::dispatchCPU->createBufferManager();

   RooArgSet serverSet;
   ::RooHelpers::getSortedComputationGraph(_topNode, serverSet);

   _evalContextCPU.resize(serverSet.size());
   if (useGPU) {
      _evalContextCUDA.resize(serverSet.size());
   }

   std::map<RooFit::Detail::DataKey, NodeInfo *> nodeInfos;

   // Fill the ordered nodes list and initialize the node info structs.
   _nodes.reserve(serverSet.size());
   std::size_t iNode = 0;
   for (RooAbsArg *arg : serverSet) {

      _nodes.emplace_back();
      auto &nodeInfo = _nodes.back();
      nodeInfo.absArg = arg;
      nodeInfo.originalOperMode = arg->operMode();
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

   if (_useGPU) {
      // create events and streams for every node
      for (auto &info : _nodes) {
         info.event = RooBatchCompute::dispatchCUDA->newCudaEvent(false);
         info.stream = RooBatchCompute::dispatchCUDA->newCudaStream();
         RooBatchCompute::Config cfg;
         cfg.setCudaStream(info.stream);
         _evalContextCUDA.setConfig(info.absArg, cfg);
      }
   }
}

/// If there are servers with the same name that got de-duplicated in the
/// `_nodes` list, we need to set their data tokens too. We find such nodes by
/// visiting the servers of every known node.
void Evaluator::syncDataTokens()
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

void Evaluator::setInput(std::string const &name, std::span<const double> inputArray, bool isOnDevice)
{
   if (isOnDevice && !_useGPU) {
      throw std::runtime_error("Evaluator can only take device array as input in CUDA mode!");
   }

   auto namePtr = RooNameReg::ptr(name.c_str());

   // Iterate over the given data spans and add them to the data map. Check if
   // they are used in the computation graph. If yes, add the span to the data
   // map and set the node info accordingly.
   std::size_t iNode = 0;
   for (auto &info : _nodes) {
      const bool fromArrayInput = info.absArg->namePtr() == namePtr;
      if (fromArrayInput) {
         info.fromArrayInput = true;
         info.absArg->setDataToken(iNode);
         info.outputSize = inputArray.size();
         if (_useGPU && info.outputSize <= 1) {
            // Empty or scalar observables from the data don't need to be
            // copied to the GPU.
            _evalContextCPU.set(info.absArg, inputArray);
            _evalContextCUDA.set(info.absArg, inputArray);
         } else if (_useGPU && info.outputSize > 1) {
            // For simplicity, we put the data on both host and device for
            // now. This could be optimized by inspecting the clients of the
            // variable.
            if (isOnDevice) {
               _evalContextCUDA.set(info.absArg, inputArray);
               auto gpuSpan = _evalContextCUDA.at(info.absArg);
               info.buffer = _bufferManager->makeCpuBuffer(gpuSpan.size());
               info.buffer->assignFromDevice(gpuSpan);
               _evalContextCPU.set(info.absArg, {info.buffer->hostReadPtr(), gpuSpan.size()});
            } else {
               _evalContextCPU.set(info.absArg, inputArray);
               auto cpuSpan = _evalContextCPU.at(info.absArg);
               info.buffer = _bufferManager->makeGpuBuffer(cpuSpan.size());
               info.buffer->assignFromHost(cpuSpan);
               _evalContextCUDA.set(info.absArg, {info.buffer->deviceReadPtr(), cpuSpan.size()});
            }
         } else {
            _evalContextCPU.set(info.absArg, inputArray);
         }
      }
      info.isDirty = !info.fromArrayInput;
      ++iNode;
   }

   _needToUpdateOutputSizes = true;
}

void Evaluator::updateOutputSizes()
{
   std::map<RooFit::Detail::DataKey, std::size_t> sizeMap;
   for (auto &info : _nodes) {
      if (info.fromArrayInput) {
         sizeMap[info.absArg] = info.outputSize;
      } else {
         // any buffer for temporary results is invalidated by resetting the output sizes
         info.buffer.reset();
      }
   }

   auto outputSizeMap =
      RooFit::BatchModeDataHelpers::determineOutputSizes(_topNode, [&](RooFit::Detail::DataKey key) -> int {
         auto found = sizeMap.find(key);
         return found != sizeMap.end() ? found->second : -1;
      });

   for (auto &info : _nodes) {
      info.outputSize = outputSizeMap.at(info.absArg);

      // In principle we don't need dirty flag propagation because the driver
      // takes care of deciding which node needs to be re-evaluated. However,
      // disabling it also for scalar mode results in very long fitting times
      // for specific models (test 14 in stressRooFit), which still needs to be
      // understood. TODO.
      if (!info.isScalar()) {
         setOperMode(info.absArg, RooAbsArg::ADirty);
      } else {
         setOperMode(info.absArg, info.originalOperMode);
      }
   }

   if (_useGPU) {
      markGPUNodes();
   }

   _needToUpdateOutputSizes = false;
}

Evaluator::~Evaluator()
{
   for (auto &info : _nodes) {
      if (!info.isVariable) {
         info.absArg->resetDataToken();
      }
   }
}

void Evaluator::computeCPUNode(const RooAbsArg *node, NodeInfo &info)
{
   using namespace Detail;

   const std::size_t nOut = info.outputSize;

   double *buffer = nullptr;
   if (nOut == 1) {
      buffer = &info.scalarBuffer;
      if (_useGPU) {
         _evalContextCUDA.set(node, {buffer, nOut});
      }
   } else {
      if (!info.hasLogged && _useGPU) {
         RooAbsArg const &arg = *info.absArg;
         oocoutI(&arg, FastEvaluations) << "The argument " << arg.ClassName() << "::" << arg.GetName()
                                        << " could not be evaluated on the GPU because the class doesn't support it. "
                                           "Consider requesting or implementing it to benefit from a speed up."
                                        << std::endl;
         info.hasLogged = true;
      }
      if (!info.buffer) {
         info.buffer = info.copyAfterEvaluation ? _bufferManager->makePinnedBuffer(nOut, info.stream)
                                                : _bufferManager->makeCpuBuffer(nOut);
      }
      buffer = info.buffer->hostWritePtr();
   }
   assignSpan(_evalContextCPU._currentOutput, {buffer, nOut});
   _evalContextCPU.set(node, {buffer, nOut});
   if (nOut > 1) {
      _evalContextCPU.enableVectorBuffers(true);
   }
   if (info.isCategory) {
      auto nodeAbsCategory = static_cast<RooAbsCategory const *>(node);
      if (nOut == 1) {
         buffer[0] = nodeAbsCategory->getCurrentIndex();
      } else {
         throw std::runtime_error("RooFit::Evaluator - non-scalar category values are not supported!");
      }
   } else {
      auto nodeAbsReal = static_cast<RooAbsReal const *>(node);
      nodeAbsReal->doEval(_evalContextCPU);
   }
   _evalContextCPU.resetVectorBuffers();
   _evalContextCPU.enableVectorBuffers(false);
   if (info.copyAfterEvaluation) {
      _evalContextCUDA.set(node, {info.buffer->deviceReadPtr(), nOut});
      if (info.event) {
         RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
      }
   }
}

/// Process a variable in the computation graph. This is a separate non-inlined
/// function such that we can see in performance profiles how long this takes.
void Evaluator::processVariable(NodeInfo &nodeInfo)
{
   RooAbsArg *node = nodeInfo.absArg;
   auto *var = static_cast<RooRealVar const *>(node);
   if (nodeInfo.lastSetValCount != var->valueResetCounter()) {
      nodeInfo.lastSetValCount = var->valueResetCounter();
      for (NodeInfo *clientInfo : nodeInfo.clientInfos) {
         clientInfo->isDirty = true;
      }
      computeCPUNode(node, nodeInfo);
      nodeInfo.isDirty = false;
   }
}

/// Process a category in the computation graph. This is a separate non-inlined
/// function such that we can see in performance profiles how long this takes.
void Evaluator::processCategory(NodeInfo &nodeInfo)
{
   RooAbsArg *node = nodeInfo.absArg;
   auto *cat = static_cast<RooAbsCategory const *>(node);
   if (nodeInfo.lastCatVal != cat->getCurrentIndex()) {
      nodeInfo.lastCatVal = cat->getCurrentIndex();
      for (NodeInfo *clientInfo : nodeInfo.clientInfos) {
         clientInfo->isDirty = true;
      }
      computeCPUNode(node, nodeInfo);
      nodeInfo.isDirty = false;
   }
}

/// Flags all the clients of a given node dirty. This is a separate non-inlined
/// function such that we can see in performance profiles how long this takes.
void Evaluator::setClientsDirty(NodeInfo &nodeInfo)
{
   for (NodeInfo *clientInfo : nodeInfo.clientInfos) {
      clientInfo->isDirty = true;
   }
}

/// Returns the value of the top node in the computation graph
std::span<const double> Evaluator::run()
{
   if (_needToUpdateOutputSizes)
      updateOutputSizes();

   ++_nEvaluations;

   if (_useGPU) {
      return getValHeterogeneous();
   }

   for (auto &nodeInfo : _nodes) {
      if (!nodeInfo.fromArrayInput) {
         if (nodeInfo.isVariable) {
            processVariable(nodeInfo);
         } else if (nodeInfo.isCategory) {
            processCategory(nodeInfo);
         } else {
            if (nodeInfo.isDirty) {
               setClientsDirty(nodeInfo);
               computeCPUNode(nodeInfo.absArg, nodeInfo);
               nodeInfo.isDirty = false;
            }
         }
      }
   }

   // return the final output
   return _evalContextCPU.at(&_topNode);
}

/// Returns the value of the top node in the computation graph
std::span<const double> Evaluator::getValHeterogeneous()
{
   for (auto &info : _nodes) {
      info.remClients = info.clientInfos.size();
      info.remServers = info.serverInfos.size();
      if (info.buffer && !info.fromArrayInput) {
         info.buffer.reset();
      }
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
         if (info.remServers == -1 && !RooBatchCompute::dispatchCUDA->cudaStreamIsActive(info.stream)) {
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

      if (!info.fromArrayInput) {
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
   return _evalContextCUDA.at(&_topNode);
}

/// Assign a node to be computed in the GPU. Scan it's clients and also assign them
/// in case they only depend on GPU nodes.
void Evaluator::assignToGPU(NodeInfo &info)
{
   using namespace Detail;

   info.remServers = -1;

   auto node = static_cast<RooAbsReal const *>(info.absArg);

   // wait for every server to finish
   for (auto *infoServer : info.serverInfos) {
      if (infoServer->event)
         RooBatchCompute::dispatchCUDA->cudaStreamWaitForEvent(info.stream, infoServer->event);
   }

   const std::size_t nOut = info.outputSize;

   double *buffer = nullptr;
   if (nOut == 1) {
      buffer = &info.scalarBuffer;
      _evalContextCPU.set(node, {buffer, nOut});
   } else {
      info.buffer = info.copyAfterEvaluation ? _bufferManager->makePinnedBuffer(nOut, info.stream)
                                             : _bufferManager->makeGpuBuffer(nOut);
      buffer = info.buffer->deviceWritePtr();
   }
   assignSpan(_evalContextCUDA._currentOutput, {buffer, nOut});
   _evalContextCUDA.set(node, {buffer, nOut});
   node->doEval(_evalContextCUDA);
   RooBatchCompute::dispatchCUDA->cudaEventRecord(info.event, info.stream);
   if (info.copyAfterEvaluation) {
      _evalContextCPU.set(node, {info.buffer->hostReadPtr(), nOut});
   }
}

/// Decides which nodes are assigned to the GPU in a CUDA fit.
void Evaluator::markGPUNodes()
{
   // Decide which nodes get evaluated on the GPU: we select nodes that support
   // CUDA evaluation and have at least one input of size greater than one.
   for (auto &info : _nodes) {
      info.computeInGPU = false;
      if (!info.absArg->canComputeBatchWithCuda()) {
         continue;
      }
      for (NodeInfo const *serverInfo : info.serverInfos) {
         if (serverInfo->outputSize > 1) {
            info.computeInGPU = true;
            break;
         }
      }
   }

   // In a second pass, figure out which nodes need to copy over their results.
   for (auto &info : _nodes) {
      info.copyAfterEvaluation = false;
      // scalar nodes don't need copying
      if (!info.isScalar()) {
         for (auto *clientInfo : info.clientInfos) {
            if (info.computeInGPU != clientInfo->computeInGPU) {
               info.copyAfterEvaluation = true;
               break;
            }
         }
      }
   }
}

/// Temporarily change the operation mode of a RooAbsArg until the
/// Evaluator gets deleted.
void Evaluator::setOperMode(RooAbsArg *arg, RooAbsArg::OperMode opMode)
{
   if (opMode != arg->operMode()) {
      _changeOperModeRAIIs.emplace(std::make_unique<ChangeOperModeRAII>(arg, opMode));
   }
}

void Evaluator::print(std::ostream &os)
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

      auto span = _evalContextCPU.at(node);

      os << "|";
      printElement(0, iNode);
      printElement(1, node->GetName());
      printElement(2, node->ClassName());
      printElement(3, nodeInfo.outputSize);
      printElement(4, nodeInfo.fromArrayInput);
      printElement(5, span[0]);

      std::cout << "\n";
   }

   printHorizontalRow();
}

/// Gets all the parameters of the RooAbsReal. This is in principle not
/// necessary, because we can always ask the RooAbsReal itself, but the
/// Evaluator has the cached information to get the answer quicker.
/// Therefore, this is not meant to be used in general, just where it matters.
/// \warning If we find another solution to get the parameters efficiently,
/// this function might be removed without notice.
RooArgSet Evaluator::getParameters() const
{
   RooArgSet parameters;
   for (auto &nodeInfo : _nodes) {
      if (nodeInfo.absArg->isFundamental()) {
         parameters.add(*nodeInfo.absArg);
      }
   }
   // Just like in RooAbsArg::getParameters(), we sort the parameters alphabetically.
   parameters.sort();
   return parameters;
}

/// \brief Sets the offset mode for evaluation.
///
/// This function sets the offset mode for evaluation to the specified mode.
/// It updates the offset mode for both CPU and CUDA evaluation contexts.
///
/// \param mode The offset mode to be set.
///
/// \note This function marks reducer nodes as dirty if the offset mode is
///       changed, because only reducer nodes can use offsetting.
void Evaluator::setOffsetMode(RooFit::EvalContext::OffsetMode mode)
{
   if (mode == _evalContextCPU._offsetMode)
      return;

   _evalContextCPU._offsetMode = mode;
   _evalContextCUDA._offsetMode = mode;

   for (auto &nodeInfo : _nodes) {
      if (nodeInfo.absArg->isReducerNode()) {
         nodeInfo.isDirty = true;
      }
   }
}

} // namespace RooFit
