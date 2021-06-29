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

#include "RooFitDriver.h"
#include "RooAbsData.h"
#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooBatchCompute.h"
#include "RooNLLVarNew.h"
#include "RooRealVar.h"
#include "RunContext.h"
#include "RooDataSet.h"

namespace ROOT {
namespace Experimental {

RooFitDriver::RooFitDriver(const RooAbsData &data, const RooNLLVarNew &topNode, int batchMode)
   : _name{topNode.GetName()}, _title{topNode.GetTitle()}, _parameters{*std::unique_ptr<RooArgSet>(
                                                              topNode.getParameters(data, true))},
     _batchMode(batchMode), _topNode(topNode), _data{&data}
{
   // fill the RunContext with the observable data and map the observables
   // by namePtr in order to replace their memory addresses later, with
   // the ones from the variables that are actually in the computation graph.
   _nEvents = data.numEntries();
   RooBatchCompute::RunContext evalData;
   data.getBatches(evalData, 0, _nEvents);
   _dataMap = evalData.spans;
   std::unordered_map<const TNamed *, const RooAbsReal *> nameResolver;
   for (auto &it : _dataMap)
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
      _dataMap[&dummy] = weights;
      nameResolver[pTNamed] = &dummy;
   }

   RooBatchCompute::dispatch = RooBatchCompute::dispatchCPU;
   // If cuda mode is on, copy all observable data to device memory
   if (_batchMode == -1) {
      if (!RooBatchCompute::dispatchCUDA)
         throw std::runtime_error(std::string("In: ") + __func__ + "(), " + __FILE__ + ":" + __LINE__ +
                                  ": Cuda implementation of the computing library is not available\n");
      RooBatchCompute::dispatch = RooBatchCompute::dispatchCUDA;
      _cudaMemDataset =
         static_cast<double *>(RooBatchCompute::dispatch->malloc(_nEvents * _dataMap.size() * sizeof(double)));
      size_t idx = 0;
      RooBatchCompute::DataMap afterCopy;
      for (auto &record : _dataMap) {
         afterCopy[record.first] = RooSpan<double>(_cudaMemDataset + idx, _nEvents);
         RooBatchCompute::dispatch->memcpyToGPU(_cudaMemDataset + idx, record.second.data(), _nEvents * sizeof(double));
         idx += _nEvents;
      }
      _dataMap.swap(afterCopy);
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

      if (pClone) // this node is an observable, update the RunContext and don't add it in `nodes`.
      {
         auto it = _dataMap.find(pClone);
         _dataMap[pAbsReal] = it->second;
         _dataMap.erase(it);

         // set nameResolver to nullptr to be able to detect future duplicates
         nameResolver[pAbsReal->namePtr()] = nullptr;
      } else // this node needs evaluation, mark it's clients
      {
         _computeQueue.push(pAbsReal);
         auto clients = pAbsReal->valueClients();
         for (auto *client : clients) {
            if (list.find(*client)) {
               ++_nodeInfos[client].nServers;
               ++_nodeInfos[pAbsReal].nClients;
            }
         }
      }
   }

   // find nodes from which we start computing the graph
   while (!_computeQueue.empty()) {
      auto node = _computeQueue.front();
      _computeQueue.pop();
      if (_nodeInfos.at(node).nServers == 0)
         _initialQueue.push(node);
   }
}

RooFitDriver::~RooFitDriver()
{
   while (!_vectorBuffers.empty()) {
      RooBatchCompute::dispatch->free(_vectorBuffers.front());
      _vectorBuffers.pop();
   }
   RooBatchCompute::dispatch->free(_cudaMemDataset);
}

double RooFitDriver::getVal()
{

   std::vector<double> nonDerivedValues;
   nonDerivedValues.reserve(_nodeInfos.size()); // to avoid reallocation

   _computeQueue = _initialQueue;
   std::unordered_map<const RooAbsArg *, NodeInfo> remaining = _nodeInfos;
   while (!_computeQueue.empty()) {
      auto node = _computeQueue.front();
      _computeQueue.pop();

      if (!node->isDerived()) {
         nonDerivedValues.push_back(node->getVal());
         _dataMap[node] = RooSpan<const double>(&nonDerivedValues.back(), 1);
      } else {

         // get an available buffer for storing the comptation results
         double *buffer = getAvailableBuffer();

         // TODO: Put integrals seperately in the computation queue
         // For now, we just assume they are scalar and assign them some temporary memory
         double normVal = 1.0;
         if (auto pAbsPdf = dynamic_cast<const RooAbsPdf *>(node)) {
            if (pAbsPdf) {
               auto integral = pAbsPdf->getIntegral(*_data->get());
               normVal = integral->getVal();
               _dataMap[integral] = RooSpan<const double>(&normVal, 1);
            }
         }

         // compute this node and register the result in the dataMap
         node->computeBatch(buffer, _nEvents, _dataMap);
         _dataMap[node] = RooSpan<const double>(buffer, _nEvents);
      }

      // update remaining of this node's clients
      // check for nodes that have now all their dependencies calculated.
      for (auto *client : node->valueClients()) {
         if (remaining.find(client) != remaining.end() && --remaining.at(client).nServers == 0)
            _computeQueue.push(static_cast<RooAbsReal *>(client));
      }
      // update remaining of this node's servers
      // check for nodes whose _vectorBuffers can now be recycled.
      for (auto *server : node->servers()) {
         if (--remaining[server].nClients == 0 && server->isDerived())
            _vectorBuffers.push(const_cast<double *>(_dataMap[static_cast<RooAbsReal *>(server)].data()));
      }
   }
   // recycle the top node's buffer and return the final value
   _vectorBuffers.push(const_cast<double *>(_dataMap[&_topNode].data()));
   return _topNode.reduce(_dataMap[&_topNode].data(), _nEvents);
}

double *RooFitDriver::getAvailableBuffer()
{
   if (_vectorBuffers.empty())
      return static_cast<double *>(RooBatchCompute::dispatch->malloc(_nEvents * sizeof(double)));
   else {
      double *buffer = _vectorBuffers.front();
      _vectorBuffers.pop();
      return buffer;
   }
}

} // namespace Experimental
} // namespace ROOT
