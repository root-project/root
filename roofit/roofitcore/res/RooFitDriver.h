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

   struct NodeInfo {
      int nServers = 0;
      int nClients = 0;
   };

private:
   double *getAvailableBuffer();

   std::string _name;
   std::string _title;
   RooArgSet _parameters;

   const int _batchMode = 0;
   double *_cudaMemDataset = nullptr;

   // used for preserving static info about the computation graph
   RooBatchCompute::DataMap _dataMap;
   const RooNLLVarNew &_topNode;
   size_t _nEvents;

   RooAbsData const *_data = nullptr;

   std::queue<const RooAbsReal *> _initialQueue;

   std::unordered_map<const RooAbsArg *, NodeInfo> _nodeInfos;
   // used for dynamically scheduling each step's computations
   std::queue<const RooAbsReal *> _computeQueue;
   std::queue<double *> _vectorBuffers;
}; // end class RooFitDriver
} // end namespace Experimental
} // end namespace ROOT

#endif // ROO_FIT_DRIVER_H
