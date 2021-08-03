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

#ifndef RooFit_RooFitDriver_h
#define RooFit_RooFitDriver_h

#include "RooBatchCompute.h"
#include "RooNLLVarNew.h"

#include <queue>
#include <unordered_map>

class RooAbsData;
class RooAbsArg;
class RooAbsReal;

namespace ROOT {
namespace Experimental {

class RooFitDriver {
public:
   RooFitDriver(const RooAbsData &data, const RooNLLVarNew &topNode, int batchMode);
   ~RooFitDriver();
   double getVal();
   RooAbsReal const &topNode() const { return _topNode; }
   std::string const &name() const { return _name; }
   std::string const &title() const { return _title; }
   RooArgSet const &parameters() const { return _parameters; }
   double errorLevel() const { return _topNode.defaultErrorLevel(); }

   class RooAbsRealWrapper final : public RooAbsReal {
   public:
      RooAbsRealWrapper() {}
      RooAbsRealWrapper(RooFitDriver &driver)
         : RooAbsReal{"RooFitDriverWrapper", "RooFitDriverWrapper"}, _driver{&driver}
      {
      }

      RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name = 0)
         : RooAbsReal{other, name}, _driver{other._driver}
      {
      }
      TObject *clone(const char *newname) const override { return new RooAbsRealWrapper(*this, newname); }

      double defaultErrorLevel() const override { return _driver->topNode().defaultErrorLevel(); }

      bool getParameters(const RooArgSet * /*observables*/, RooArgSet &outputSet,
                         bool /*stripDisconnected=true*/) const override
      {
         outputSet.add(_driver->parameters());
         return false;
      }

      double getValV(const RooArgSet *) const override { return evaluate(); }

   protected:
      double evaluate() const override { return _driver ? _driver->getVal() : 0.0; }

   private:
      RooFitDriver *_driver = nullptr;
   };

   std::unique_ptr<RooAbsReal> makeAbsRealWrapper()
   {
      return std::unique_ptr<RooAbsReal>{new RooAbsRealWrapper{*this}};
   }

private:
   struct NodeInfo {
      int nClients = 0;
      int nServers = 0;
      int remClients = 0;
      int remServers = 0;
      bool computeInScalarMode = false;
      bool computeInGPU = false;
      bool copyAfterEvaluation = false;
      cudaStream_t *stream = nullptr;
      cudaEvent_t *event = nullptr;
      ~NodeInfo()
      {
         if (computeInGPU) {
            RooBatchCompute::dispatchCUDA->deleteCudaEvent(event);
            RooBatchCompute::dispatchCUDA->deleteCudaStream(stream);
         }
      }
   };
   void updateMyClients(const RooAbsReal *node);
   void updateMyServers(const RooAbsReal *node);
   void handleIntegral(const RooAbsReal *node);
   void markGPUNodes();
   void assignToGPU(const RooAbsReal *node);
   double *getAvailableCPUBuffer();
   double *getAvailableGPUBuffer();
   double *getAvailablePinnedBuffer();

   std::string _name;
   std::string _title;
   RooArgSet _parameters;

   const int _batchMode = 0;
   double *_cudaMemDataset;

   // used for preserving static info about the computation graph
   RooBatchCompute::DataMap _dataMapCPU;
   RooBatchCompute::DataMap _dataMapCUDA;
   const RooNLLVarNew &_topNode;
   const RooAbsData *const _data = nullptr;
   const size_t _nEvents;
   std::unordered_map<const RooAbsReal *, NodeInfo> _nodeInfos;

   // used for preserving resources
   std::queue<double *> _cpuBuffers;
   std::queue<double *> _gpuBuffers;
   std::queue<double *> _pinnedBuffers;
   std::vector<double> _nonDerivedValues;
}; // end class RooFitDriver
} // end namespace Experimental
} // end namespace ROOT

#endif // ROO_FIT_DRIVER_H
