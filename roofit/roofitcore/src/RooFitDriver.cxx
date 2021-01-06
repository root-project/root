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

namespace ROOT {
namespace Experimental {

RooFitDriver::RooFitDriver(const RooAbsData &data, const RooNLLVarNew &topNode, int batchMode)
   : _name{topNode.GetName()}, _title{topNode.GetTitle()}, _parameters{*std::unique_ptr<RooArgSet>(
                                                              topNode.getParameters(data, true))},
     _batchMode(batchMode), _topNode(topNode)
{
   // get a set with all the observables as the normalization set
   // and call getVal to trigger the creation of the integrals.
   _observables = data.get();
   _topNode.getPdf()->getVal(_observables);

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

   RooBatchCompute::dispatch = RooBatchCompute::dispatchCPU;
   // If cuda mode is on, copy all observable data to device memory
   if (_batchMode == -1) {
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
      auto pRealVar = dynamic_cast<RooRealVar *>(&list[i]);

      if (pClone) // this node is an observable, update the RunContext and don't add it in `nodes`.
      {
         auto it = _dataMap.find(pClone);
         _dataMap[pAbsReal] = it->second;
         _dataMap.erase(it);

         // set nameResolver to nullptr to be able to detect future duplicates
         nameResolver[pAbsReal->namePtr()] = nullptr;
      } else if (!pRealVar) // this node needs computing, mark it's clients
      {
         _computeQueue.push(pAbsReal);
         auto clients = pAbsReal->valueClients();
         for (auto *client : clients)
            ++_nServersClients[static_cast<const RooAbsReal *>(client)].first;
         _nServersClients[pAbsReal].second = clients.size();
      } else // this node is a scalar parameter
      {
         _dataMap.emplace(pAbsReal, RooSpan<const double>(pRealVar->getValPtr(), 1));
         pRealVar->setError(0.0);
      }
   }

   // find nodes from which we start computing the graph
   while (!_computeQueue.empty()) {
      auto node = _computeQueue.front();
      _computeQueue.pop();
      if (_nServersClients.at(node).first == 0)
         _initialQueue.push(node);
   }
}

RooFitDriver::~RooFitDriver()
{
   while (!_buffers.empty()) {
      RooBatchCompute::dispatch->free(_buffers.front());
      _buffers.pop();
   }
   RooBatchCompute::dispatch->free(_cudaMemDataset);
}

double RooFitDriver::getVal()
{
   _computeQueue = _initialQueue;
   _nRemainingServersClients = _nServersClients;
   while (!_computeQueue.empty()) {
      auto node = _computeQueue.front();
      _computeQueue.pop();

      // get an available buffer for storing the comptation results
      double *buffer;
      if (_buffers.empty())
         buffer = static_cast<double *>(RooBatchCompute::dispatch->malloc(_nEvents * sizeof(double)));
      else {
         buffer = _buffers.front();
         _buffers.pop();
      }

      // TODO: Put integrals seperately in the computation queue
      // For now, we just assume they are scalar and assign them some temporary memory
      double normVal = 1.0;
      auto pAbsPdf = dynamic_cast<const RooAbsPdf *>(node);
      if (pAbsPdf) {
         normVal = pAbsPdf->getIntegral()->getVal();
         _dataMap[pAbsPdf->getIntegral()] = RooSpan<const double>(&normVal, 1);
      }

      // compute this node and register the result in the dataMap
      node->computeBatch(buffer, _nEvents, _dataMap);
      _dataMap[node] = RooSpan<const double>(buffer, _nEvents);

      // update _nRemainingServersClients of this node's clients
      // check for nodes that have now all their dependencies calculated.
      for (auto *pAbsArg : node->valueClients()) {
         auto client = static_cast<const RooAbsReal *>(pAbsArg);
         if (--_nRemainingServersClients.at(client).first == 0)
            _computeQueue.push(client);
      }
      // update _nRemainingServersClients of this node's servers
      // check for nodes whose buffers can now be recycled.
      for (auto *pAbsArg : node->servers()) {
         auto server = static_cast<const RooAbsReal *>(pAbsArg);
         if (--_nRemainingServersClients[server].second == 0)
            _buffers.push(const_cast<double *>(_dataMap[server].data()));
      }
   }
   // recycle the top node's buffer and return the final value
   _buffers.push(const_cast<double *>(_dataMap[&_topNode].data()));
   return _topNode.reduce(_dataMap[&_topNode].data(), _nEvents);
}

} // namespace Experimental
} // namespace ROOT
