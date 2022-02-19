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
it is being used for evaluating a RooAbsArg object and supplying the
value to the minimizer, during a fit. The class scans the dependencies and
schedules the computations in a secure and efficient way. The computations
take place in the RooBatchCompute library and can be carried off by
either the cpu or a cuda gpu; this class takes care of data transfers.
An instance of this class is created every time RooAbsPdf::fitTo() is called
and gets destroyed when the fitting ends.
**/

#include <RooFitDriver.h>

#include <RooAbsCategory.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooAbsPdf.h>
#include <RooArgList.h>
#include <RooBatchCompute.h>
#include <RooDataSet.h>
#include <RooMsgService.h>
#include <RooNLLVarNew.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooBatchCompute/Initialisation.h>

#include <ROOT/StringUtils.hxx>

#include <iomanip>
#include <numeric>
#include <thread>

namespace ROOT {
namespace Experimental {

using namespace Detail;
using namespace std::chrono;

RooFitDriver::Dataset::Dataset(RooBatchCompute::RunContext const &runContext)
{
   for (auto const &item : runContext.spans) {
      _dataSpans[item.first->namePtr()] = item.second;
      _nEvents = std::max(_nEvents, item.second.size());
   }
}

RooFitDriver::Dataset::Dataset(RooAbsData const &data, RooArgSet const &observables, std::string_view rangeName,
                               RooAbsCategory const *indexCat)
   : _nEvents{static_cast<size_t>(data.numEntries())}
{

   // fill the RunContext with the observable data and map the observables
   // by namePtr in order to replace their memory addresses later, with
   // the ones from the variables that are actually in the computation graph.
   RooBatchCompute::RunContext evalData;
   data.getBatches(evalData, 0, _nEvents);
   for (auto const &item : evalData.spans) {
      _dataSpans[item.first->namePtr()] = item.second;
   }

   // Get the category batches and cast the also to double branches to put in
   // the data map
   for (auto const &item : data.getCategoryBatches(0, _nEvents)) {

      const TNamed *namePtr = RooNameReg::instance().constPtr(item.first.c_str());
      RooSpan<const RooAbsCategory::value_type> intSpan{item.second};

      _buffers.emplace(_nEvents);
      double *buffer = _buffers.top().data();

      for (std::size_t i = 0; i < _nEvents; ++i) {
         buffer[i] = static_cast<double>(intSpan[i]);
      }
      _dataSpans[namePtr] = RooSpan<const double>(buffer, _nEvents);
   }

   // Check if there is a batch for weights and if it's already in the dataMap.
   // If not, we need to put the batch and give as a key a RooRealVar* that has
   // the same name as RooNLLVarNew's _weight proxy, so that it gets renamed like
   // every other observable.
   RooSpan<const double> weights = data.getWeightBatch(0, _nEvents);
   if (!weights.empty()) {
      std::string weightVarName = "_weight";
      if (auto *dataSet = dynamic_cast<RooDataSet const *>(&data)) {
         if (dataSet->weightVar())
            weightVarName = dataSet->weightVar()->GetName();
      }
      const TNamed *pTNamed = RooNameReg::instance().constPtr(weightVarName.c_str());
      _dataSpans[pTNamed] = weights;
   }

   // Now we have do do the range selection
   if (!rangeName.empty()) {
      // figure out which events are in the range
      std::vector<bool> isInRange(_nEvents, false);
      for (auto const &range : ROOT::Split(rangeName, ",")) {
         std::vector<bool> isInSubRange(_nEvents, true);
         for (auto *observable : static_range_cast<RooAbsRealLValue *>(observables)) {
            RooSpan<const double> allValues = _dataSpans[observable->namePtr()];
            observable->inRange({allValues.data(), _nEvents}, range, isInSubRange);
         }
         for (std::size_t i = 0; i < isInSubRange.size(); ++i) {
            isInRange[i] = isInRange[i] | isInSubRange[i];
         }
      }

      // reset the number of events
      _nEvents = std::accumulate(isInRange.begin(), isInRange.end(), 0);

      // do the data reduction in the data map
      for (auto const &item : _dataSpans) {
         auto const &allValues = item.second;
         _buffers.emplace(_nEvents);
         double *buffer = _buffers.top().data();
         std::size_t j = 0;
         for (std::size_t i = 0; i < isInRange.size(); ++i) {
            if (isInRange[i]) {
               buffer[j] = allValues[i];
               ++j;
            }
         }
         _dataSpans[item.first] = RooSpan<const double>{buffer, _nEvents};
      }
   }

   if (indexCat) {
      splitByCategory(*indexCat);
   }
}

void RooFitDriver::Dataset::splitByCategory(RooAbsCategory const &category)
{

   std::stack<std::vector<double>> oldBuffers;
   std::swap(_buffers, oldBuffers);

   auto catVals = _dataSpans[category.namePtr()];

   std::map<const TNamed *, RooSpan<const double>> dataMapSplit;

   for (auto const &dataMapItem : _dataSpans) {

      auto const &varNamePtr = dataMapItem.first;
      if (varNamePtr == category.namePtr())
         continue;

      std::map<RooAbsCategory::value_type, std::vector<double>> valuesMap;

      auto xVals = _dataSpans[varNamePtr];
      for (std::size_t i = 0; i < xVals.size(); ++i) {
         valuesMap[catVals[i]].push_back(xVals[i]);
      }

      for (auto const &item : valuesMap) {
         RooAbsCategory::value_type index = item.first;
         auto variableName = std::string("_") + category.lookupName(index) + "_" + varNamePtr->GetName();
         auto variableNamePtr = RooNameReg::instance().constPtr(variableName.c_str());

         _buffers.emplace(std::move(item.second));
         auto const &values = _buffers.top();
         dataMapSplit[variableNamePtr] = RooSpan<const double>(values.data(), values.size());
      }
   }

   _dataSpans = std::move(dataMapSplit);
}

/**
Construct a new RooFitDriver. The constructor analyzes and saves metadata about the graph,
useful for the evaluation of it that will be done later. In case cuda mode is selected,
there's also some cuda-related initialization.

\param data The dataset for the fitting model
\param topNode The RooNLLVaNew object that sits on top of the graph and whose value the minimiser needs.
\param observables The observabes of the pdf
\param normSet
\param batchMode The computation mode of the RooBatchCompute library, accepted values are `RooBatchCompute::Cpu` and
`RooBatchCompute::Cuda`.
\param rangeName the range name
\param indexCat
**/
RooFitDriver::RooFitDriver(const RooAbsData &data, const RooAbsReal &topNode, RooArgSet const &normSet,
                           RooFit::BatchModeOption batchMode, std::string_view rangeName,
                           RooAbsCategory const *indexCat)
   : _name{topNode.GetName()}, _title{topNode.GetTitle()}, _parameters{*std::unique_ptr<RooArgSet>(
                                                              topNode.getParameters(*data.get(), true))},
     _batchMode{batchMode}, _dataset{data, *std::unique_ptr<RooArgSet>(topNode.getObservables(data)), rangeName,
                                     indexCat},
     _topNode{topNode}, _normSet{std::make_unique<RooArgSet>(normSet)}
{
   init();
}

RooFitDriver::RooFitDriver(RooBatchCompute::RunContext const &data, const RooAbsReal &topNode, RooArgSet const &normSet)
   : _name{topNode.GetName()}, _title{topNode.GetTitle()},
     _batchMode{RooFit::BatchModeOption::Cpu}, _dataset{data}, _topNode{topNode}, _normSet{std::make_unique<RooArgSet>(normSet)}
{
   init();
}

void RooFitDriver::init()
{
   // Initialize RooBatchCompute
   RooBatchCompute::init();

   // Some checks and logging of used architectures
   {
      auto log = [](std::string_view message) {
         oocxcoutI(static_cast<RooAbsArg *>(nullptr), FastEvaluations) << message << std::endl;
      };

      if (_batchMode == RooFit::BatchModeOption::Cuda && !RooBatchCompute::dispatchCUDA) {
         throw std::runtime_error(std::string("In: ") + __func__ + "(), " + __FILE__ + ":" + __LINE__ +
                                  ": Cuda implementation of the computing library is not available\n");
      }
      if (RooBatchCompute::dispatchCPU->architecture() == RooBatchCompute::Architecture::GENERIC) {
         log("using generic CPU library compiled with no vectorizations");
      } else {
         log(std::string("using CPU computation library compiled with -m") +
             RooBatchCompute::dispatchCPU->architectureName());
      }
      if (_batchMode == RooFit::BatchModeOption::Cuda) {
         log("using CUDA computation library");
      }
   }

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

   for (RooAbsArg *arg : serverSet) {
      _orderedNodes.push_back(arg);
      auto &argInfo = _nodeInfos[arg];
      if (_dataset.contains(arg)) {
         _dataMapCPU[arg] = _dataset.span(arg);
         argInfo.outputSize = _dataset.span(arg).size();
      }

      for (auto *client : arg->clients()) {
         // we use containsInstance instead of find to match by pointer and not name
         if (!serverSet.containsInstance(*client))
            continue;

         auto &clientInfo = _nodeInfos[client];

         ++clientInfo.nServers;
         ++argInfo.nClients;
      }
   }

   // Sort the nodes for good
   std::unordered_set<TNamed const *> seenNodes;
   for (std::size_t iNode = 0; iNode < _orderedNodes.size(); ++iNode) {
      RooAbsArg *node = _orderedNodes[iNode];
      bool movedNode = false;
      for (RooAbsArg *server : node->servers()) {
         if (server->isValueServer(*node) && seenNodes.find(server->namePtr()) == seenNodes.end()) {
            auto found = std::find_if(_orderedNodes.begin(), _orderedNodes.end(),
                                      [server](RooAbsArg *arg) { return arg->namePtr() == server->namePtr(); });
            if (found == _orderedNodes.end()) {

               throw std::runtime_error(std::string("Node ") + node->GetName() + " depends on " + server->GetName() +
                                        ", but this node is missing in the computation queue!");
            }
            _orderedNodes.erase(found);
            _orderedNodes.insert(_orderedNodes.begin() + iNode, server);
            movedNode = true;
            break;
         }
      }
      if (movedNode) {
         --iNode;
         continue;
      }
      seenNodes.insert(node->namePtr());
   }

   determineOutputSizes(serverSet);

   for (auto *arg : serverSet) {
      auto &info = _nodeInfos[arg];

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

   // copy observable data to the gpu
   // TODO: use separate buffers here
   _cudaMemDataset = static_cast<double *>(
      RooBatchCompute::dispatchCUDA->cudaMalloc(_dataset.totalNumberOfEntries() * sizeof(double)));
   size_t idx = 0;
   for (auto &record : _dataMapCPU) {
      std::size_t size = record.second.size();
      _dataMapCUDA[record.first] = RooSpan<double>(_cudaMemDataset + idx, size);
      RooBatchCompute::dispatchCUDA->memcpyToCUDA(_cudaMemDataset + idx, record.second.data(), size * sizeof(double));
      idx += size;
   }

   // create events and streams for every node
   for (auto &item : _nodeInfos) {
      item.second.event = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
      item.second.eventStart = RooBatchCompute::dispatchCUDA->newCudaEvent(true);
      item.second.stream = RooBatchCompute::dispatchCUDA->newCudaStream();
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
   if (_nodeInfos.at(&_topNode).computeInGPU) {
      double *buffer = new double[_dataset.size()];
      RooBatchCompute::dispatchCUDA->memcpyToCPU(buffer, _dataMapCPU.at(&_topNode).data(),
                                                 _dataset.size() * sizeof(double));
      _dataMapCPU[&_topNode] = RooSpan<const double>(buffer, _dataset.size());
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

   if (_batchMode == RooFit::BatchModeOption::Cuda)
      return getValHeterogeneous();

   for (std::size_t iNode = 0; iNode < _orderedNodes.size(); ++iNode) {
      RooAbsArg *node = _orderedNodes[iNode];
      if (!_dataset.contains(node)) {
         computeCPUNode(node, _nodeInfos[node]);
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
         assignToGPU(it.first);

   int nNodes = _nodeInfos.size();
   while (nNodes) {
      // find finished gpu nodes
      if (_batchMode == RooFit::BatchModeOption::Cuda)
         for (auto &it : _nodeInfos)
            if (it.second.remServers == -1 && !RooBatchCompute::dispatchCUDA->streamIsActive(it.second.stream)) {
               if (_getValInvocations == 2) {
                  float ms = RooBatchCompute::dispatchCUDA->cudaEventElapsedTime(it.second.eventStart, it.second.event);
                  it.second.cudaTime += microseconds{int(1000.0 * ms)};
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
         std::this_thread::sleep_for(milliseconds(1));
         continue;
      }

      // compute next cpu node
      RooAbsArg const *node = it->first;
      NodeInfo &info = it->second;
      info.remServers = -2; // so that it doesn't get picked again
      nNodes--;

      if (!_dataset.contains(node)) {
         computeCPUNode(node, info);
      }
      updateMyClients(node);
      updateMyServers(node);
   } // while (nNodes)

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
   updateMyClients(node);
}

/// Check the clients of a node that has been computed and assign them to gpu
/// if they are ready to be computed
void RooFitDriver::updateMyClients(RooAbsArg const *node)
{
   NodeInfo &info = _nodeInfos.at(node);
   for (auto *client : node->clients()) {
      if (_nodeInfos.count(client) == 0)
         continue; // client not part of the computation graph
      NodeInfo &infoClient = _nodeInfos.at(client);

      if (info.remServers == -1 && infoClient.computeInGPU && --infoClient.remServers == 0)
         assignToGPU(client); // updateMyCilents called when assigning to gpu
      else if (info.remServers == -2 && info.computeInGPU && !infoClient.computeInGPU)
         --infoClient.remServers; // updateMyClients called after finishing a gpu node
      else if (!info.computeInGPU && --infoClient.remServers == 0 && infoClient.computeInGPU)
         assignToGPU(client); // updateMyClients called after finishing a cpu node
   }
}

/// Check the servers of a node that has been computed and release it's resources
/// if they are no longer needed
void RooFitDriver::updateMyServers(const RooAbsArg *node)
{
   for (auto *server : node->servers()) {
      if (_nodeInfos.count(server)) {
         _nodeInfos.at(server).decrementRemainingClients();
      }
   }
}

/// Measure the time for transfering data from host to device and vice-versa
std::pair<microseconds, microseconds> RooFitDriver::memcpyBenchmark()
{
   // For the number of bytes in this benchmark we take the average size of
   // spans in the dataset.
   std::size_t nBytes = _dataset.totalNumberOfEntries() * sizeof(double) / _dataset.spans().size();

   std::pair<microseconds, microseconds> ret;
   auto hostArr = static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMallocHost(nBytes));
   auto deviArr = static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMalloc(nBytes));
   for (int i = 0; i < 5; i++) {
      auto start = steady_clock::now();
      RooBatchCompute::dispatchCUDA->memcpyToCUDA(deviArr, hostArr, nBytes);
      ret.first += duration_cast<microseconds>(steady_clock::now() - start);
      start = steady_clock::now();
      RooBatchCompute::dispatchCUDA->memcpyToCPU(hostArr, deviArr, nBytes);
      ret.second += duration_cast<microseconds>(steady_clock::now() - start);
   }
   RooBatchCompute::dispatchCUDA->cudaFreeHost(hostArr);
   RooBatchCompute::dispatchCUDA->cudaFree(deviArr);
   ret.first /= 5;
   ret.second /= 5;
   return ret;
}

/// This methods simulates the computation of the whole graph and the time it takes
/// and decides what to compute in gpu. The decision is made on the basis of avoiding
/// leaving either the gpu or the cpu idle at any time, if possible, and on assigning
/// to each piece of hardware a computation that is significantly slower on the other part.
/// The nodes may be assigned to the non-efficient side (cpu or gpu) to prevent idleness
/// only if the absolute difference cpuTime-cudaTime does not exceed the diffThreshold.
microseconds RooFitDriver::simulateFit(microseconds h2dTime, microseconds d2hTime, microseconds diffThreshold)
{
   int nNodes = _nodeInfos.size();
   // launch scalar nodes (assume they are computed in 0 time)
   for (auto &it : _nodeInfos)
      if (it.second.computeInScalarMode) {
         nNodes--;
         it.second.timeLaunched = microseconds{0};
      } else
         it.second.timeLaunched = microseconds{-1};

   RooAbsArg const *cpuNode = nullptr;
   RooAbsArg const *cudaNode = nullptr;
   microseconds simulatedTime{0};
   while (nNodes) {
      microseconds minDiff = microseconds::max(), maxDiff = -minDiff; // diff = cpuTime - cudaTime
      RooAbsArg const *cpuCandidate = nullptr;
      RooAbsArg const *cudaCandidate = nullptr;
      microseconds cpuDelay, cudaDelay;
      for (auto &it : _nodeInfos) {
         if (it.second.timeLaunched >= microseconds{0})
            continue; // already launched
         microseconds diff{it.second.cpuTime - it.second.cudaTime}, cpuWait{0}, cudaWait{0};

         for (auto *server : it.first->servers()) {
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
            cpuCandidate = it.first;
         }
         if (diff > maxDiff && it.first->canComputeBatchWithCuda()) {
            maxDiff = diff;
            cudaDelay = cudaWait;
            cudaCandidate = it.first;
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
         etaCPU = _nodeInfos[cpuNode].timeLaunched + _nodeInfos[cpuNode].cpuTime;
      if (cudaNode)
         etaCUDA = _nodeInfos[cudaNode].timeLaunched + _nodeInfos[cudaNode].cudaTime;
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
   if (_getValInvocations == 1)
      return;                        // leave everything to be computed (and timed) in cpu
   else if (_getValInvocations == 2) // compute (and time) as much as possible in gpu
      for (auto &item : _nodeInfos)
         item.second.computeInGPU = !item.second.computeInScalarMode && item.first->canComputeBatchWithCuda();
   else // assign nodes to gpu using a greedy algorithm
   {
      auto transferTimes = memcpyBenchmark();
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

   for (auto &item : _nodeInfos)
      if (!item.second.computeInScalarMode) // scalar nodes don't need copying
         for (auto *client : item.first->clients()) {
            if (_nodeInfos.count(client) == 0)
               continue;
            auto &info = _nodeInfos.at(client);
            if (item.second.computeInGPU != info.computeInGPU) {
               item.second.copyAfterEvaluation = true;
               break;
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
            << std::setw(20) << item.first->GetName() << "\t" << item.first << "\t"
            << (item.second.computeInGPU ? "CUDA" : "CPU") << "\t" << item.second.cpuTime.count() << "us\t"
            << item.second.cudaTime.count() << "us\n";
   }
}

void RooFitDriver::determineOutputSizes(RooArgSet const &serverSet)
{
   for (auto *arg : _orderedNodes) {
      auto &argInfo = _nodeInfos[arg];
      for (auto *client : arg->valueClients()) {
         if (serverSet.containsInstance(*client)) {
            auto &clientInfo = _nodeInfos[client];
            if (!client->isReducerNode()) {
               clientInfo.outputSize = std::max(clientInfo.outputSize, argInfo.outputSize);
            }
         }
      }
   }
}

} // namespace Experimental
} // namespace ROOT
