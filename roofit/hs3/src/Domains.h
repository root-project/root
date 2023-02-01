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
#include <unordered_map>
#include <vector>

class RooRealVar;

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
   void readVariable(const char *name, double min, double max);
   void readVariable(RooRealVar const &);
   void writeVariable(RooRealVar &) const;

   void readJSON(RooFit::Detail::JSONNode const &);
   void writeJSON(RooFit::Detail::JSONNode &) const;

private:
   class ProductDomain {
   public:
      void readVariable(const char *name, double min, double max);
      void writeVariable(RooRealVar &) const;

      void readJSON(RooFit::Detail::JSONNode const &);
      void writeJSON(RooFit::Detail::JSONNode &) const;

   private:
      struct ProductDomainElement {
         bool hasMin = false;
         bool hasMax = false;
         double min = 0.0;
         double max = 0.0;
      };

      std::unordered_map<std::string, ProductDomainElement> _map;
   };

   std::unordered_map<std::string, ProductDomain> _map;
};

} // namespace Detail
} // namespace JSONIO
} // namespace RooFit

#endif
