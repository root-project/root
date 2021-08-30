#include "RooFitDriver.h"
#include "RooAbsData.h"
#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooBatchCompute.h"
#include "RooNLLVarNew.h"
#include "RooRealVar.h"
#include "RunContext.h"

RooFitDriver::RooFitDriver(const RooAbsData& data, const RooNLLVarNew& _topNode, int _batchMode)
  : _name{_topNode.GetName()}, _title{_topNode.GetTitle()}
  , _parameters{*std::unique_ptr<RooArgSet>(_topNode.getParameters(data, true))}
  , batchMode(_batchMode), topNode(_topNode)
{
  // get a set with all the observables as the normalization set
  // and call getVal to trigger the creation of the integrals.
  observables = data.get();
  topNode.getPdf()->getVal(observables);

  // fill the RunContext with the observable data and map the observables
  // by namePtr in order to replace their memory addresses later, with
  // the ones from the variables that are actually in the computation graph. 
  nEvents = data.numEntries();
  rbc::RunContext evalData;
  data.getBatches(evalData, 0, nEvents);
  dataMap = evalData.spans;
  std::unordered_map<const TNamed*,const RooAbsReal*> nameResolver;
  for (auto& it:dataMap) nameResolver[it.first->namePtr()]=it.first;

  // Check if there is a batch for weights and if it's already in the dataMap.
  // If not, we need to put the batch and give as a key a RooRealVar* that has
  // the same name as RooNLLVarNew's _weight proxy, so that it gets renamed like
  // every other observable.
  RooSpan<const double> weights = data.getWeightBatch(0, nEvents);
  std::string weightVarName = data.getWeightVarName()!="" ? data.getWeightVarName() : "_weight";
  RooRealVar dummy (weightVarName.c_str(), "dummy", 0.0);
  const TNamed* pTNamed = dummy.namePtr();
  if (!weights.empty() && nameResolver.count(pTNamed)==0)
  {
    dataMap[&dummy] = weights;
    nameResolver[pTNamed] = &dummy;
  }

  rbc::dispatch = rbc::dispatch_cpu;
  // If cuda mode is on, copy all observable data to device memory
  if (batchMode == -1)
  {
    if (!rbc::dispatch_gpu) 
      throw std::runtime_error(std::string("In: ")+__func__+"(), "+__FILE__+":"+__LINE__+": Cuda implementation of the computing library is not available\n");
    rbc::dispatch = rbc::dispatch_gpu;
    cudaMemDataset = static_cast<double*>(rbc::dispatch->malloc( nEvents*dataMap.size()*sizeof(double) ));
    size_t idx=0;
    rbc::DataMap afterCopy;
    for (auto& record:dataMap)
    {
      afterCopy[record.first] = RooSpan<double>(cudaMemDataset+idx, nEvents);
      rbc::dispatch->memcpyToGPU(cudaMemDataset+idx, record.second.data(), nEvents*sizeof(double));
      idx += nEvents;
    }
    dataMap.swap(afterCopy);
  }

  // Get a serial list of the nodes in the computation graph.
  // treeNodeServelList() is recursive and adds the top node before the children,
  // so reversing the list gives us a topological ordering of the graph.
  RooArgList list;
  topNode.treeNodeServerList(&list);
  for (int i=list.size()-1; i>=0; i--)
  {
    auto pAbsReal = dynamic_cast<RooAbsReal*>(&list[i]);
    if (!pAbsReal) continue;
    const bool alreadyExists = nameResolver.count(pAbsReal->namePtr());
    const RooAbsReal* pClone = nameResolver[pAbsReal->namePtr()];
    if (alreadyExists && !pClone) continue; // node included multiple times in the list
    auto pRealVar = dynamic_cast<RooRealVar*>(&list[i]);
      
    if (pClone) //this node is an observable, update the RunContext and don't add it in `nodes`.
    {
      auto it = dataMap.find(pClone);
      dataMap[pAbsReal]=it->second;
      dataMap.erase(it);

      // set nameResolver to nullptr to be able to detect future duplicates
      nameResolver[pAbsReal->namePtr()] = nullptr;
    }
    else if (pRealVar) //this node is a scalar parameter
    {
      dataMap.emplace( pAbsReal, RooSpan<const double>(pRealVar->getValPtr(),1) );
      pRealVar->setError(0.0);
    }
    else //this node needs computing, mark it's clients
    {
      computeQueue.push(pAbsReal);
      auto clients = pAbsReal->valueClients();
      for (auto* client:clients) {
        if(list.find(*client)) {
          ++nServersClients[static_cast<const RooAbsReal*>(client)].first;
        }
      }
      nServersClients[pAbsReal].second = clients.size();
    }
  }

  // find nodes from which we start computing the graph
  while (!computeQueue.empty())
  {
    auto node = computeQueue.front();
    computeQueue.pop();
    if (nServersClients.at(node).first == 0)
      initialQueue.push(node);
  }
}

RooFitDriver::~RooFitDriver()
{
  while (!buffers.empty())
  {
    rbc::dispatch->free( buffers.front() );
    buffers.pop();
  }
  rbc::dispatch->free(cudaMemDataset);
}

double RooFitDriver::getVal()
{
  computeQueue = initialQueue;
  nRemainingServersClients = nServersClients;
  while (!computeQueue.empty())
  {
    auto node = computeQueue.front();
    computeQueue.pop();

    // get an available buffer for storing the comptation results
    double* buffer;
    if (buffers.empty())
      buffer = static_cast<double*>(rbc::dispatch->malloc( nEvents*sizeof(double) ));
    else
    {
      buffer = buffers.front();
      buffers.pop();
    }

    // TODO: Put integrals seperately in the computation queue
    // For now, we just assume they are scalar and assign them some temporary memory
    double normVal=1.0;
    auto pAbsPdf = dynamic_cast<const RooAbsPdf*>(node);
    if (pAbsPdf)
    {
      normVal = pAbsPdf->getIntegral()->getVal();
      dataMap[pAbsPdf->getIntegral()] = RooSpan<const double>(&normVal,1);
    }

    // compute this node and register the result in the dataMap
    node->computeBatch(buffer, nEvents, dataMap);
    dataMap[node] = RooSpan<const double>(buffer,nEvents);

    // update nRemainingServersClients of this node's clients
    // check for nodes that have now all their dependencies calculated.
    for (auto* pAbsArg:node->valueClients())
    {
      auto client = static_cast<const RooAbsReal*>(pAbsArg);
      if (nRemainingServersClients.find(client) != nRemainingServersClients.end()
              && --nRemainingServersClients.at(client).first == 0)
        computeQueue.push(client);
    }
    // update nRemainingServersClients of this node's servers
    // check for nodes whose buffers can now be recycled.
    for (auto* pAbsArg:node->servers())
    {
      auto server = static_cast<const RooAbsReal*>(pAbsArg);
      if (--nRemainingServersClients[server].second == 0)
        buffers.push(const_cast<double*>( dataMap[server].data() ));
    }
  }
  // recycle the top node's buffer and return the final value
  buffers.push(const_cast<double*>( dataMap[&topNode].data() ));
  return topNode.reduce(dataMap[&topNode].data(), nEvents);
}
