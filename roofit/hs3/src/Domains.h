/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Jan 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_JSONIO_Detail_Domains_h
#define RooFit_JSONIO_Detail_Domains_h

#include <string>
#include <map>
#include <vector>

class RooAbsBinning;
class RooRealVar;
class RooWorkspace;

namespace RooFit {
namespace Detail {
class JSONNode;
class JSONTree;
} // namespace Detail
} // namespace RooFit

namespace RooFit {
namespace JSONIO {
namespace Detail {

class Domains {
public:
   void readVariable(const char *name, double min, double max, const char *domain);
   void readVariable(const char *name, double min, double max);

   void readVariable(RooRealVar const &);
   void writeVariable(RooRealVar &) const;

   void readJSON(RooFit::Detail::JSONNode const &);
   void writeJSON(RooFit::Detail::JSONNode &) const;

   bool hasVariable(const char *name) const;

   void populate(RooWorkspace &ws) const;

   class ProductDomain {
   public:
      void readVariable(const RooRealVar &);
      void readVariable(const char *name, RooAbsBinning const &binning);
      void readVariable(const char *name, double min, double max);
      void writeVariable(RooRealVar &) const;

      void readJSON(RooFit::Detail::JSONNode const &);
      void writeJSON(RooFit::Detail::JSONNode &) const;

      bool hasVariable(const char *name) const;

      void populate(RooWorkspace &ws) const;
      void registerBinnings(const char *name, RooWorkspace &ws) const;

   private:
      struct ProductDomainElement {
         bool hasMin = false;
         bool hasMax = false;
         double min = 0.0;
         double max = 0.0;
         bool hasNBins = false;
         int nBins = 0;
         std::vector<double> edges;
      };

      static void applyBinning(RooRealVar &var, ProductDomainElement const &elem, const char *name = nullptr);
      static void readBinning(ProductDomainElement &elem, RooAbsBinning const &binning);
      static void writeBinning(RooFit::Detail::JSONNode &node, ProductDomainElement const &elem);

      std::map<std::string, ProductDomainElement> _map;
   };

   std::map<std::string, ProductDomain> _map;
};

} // namespace Detail
} // namespace JSONIO
} // namespace RooFit

#endif
