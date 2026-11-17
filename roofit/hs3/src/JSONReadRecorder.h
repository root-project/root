/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Nov 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_JSONReadRecorder_h
#define RooFit_Detail_JSONReadRecorder_h

#include <RooFit/Detail/JSONInterface.h>

#include <set>
#include <string>
#include <vector>

namespace RooFit {
namespace Detail {

/// Records which components of a JSON map are read while it is alive, through any node of the tree. Like that, the
/// components that no importer understands can be reported without any change to the importers. Iterating over the
/// components or writing the map counts as reading all of them.
class JSONReadRecorder {
public:
   explicit JSONReadRecorder(JSONNode const &node);
   ~JSONReadRecorder();
   JSONReadRecorder(JSONReadRecorder const &) = delete;
   JSONReadRecorder &operator=(JSONReadRecorder const &) = delete;

   /// The components of the map that were not read.
   std::vector<std::string> unreadComponents() const;

   /// Called by the tree for every read of the component `key` of `value`, where nullptr means all components.
   void record(void const *value, std::string const *key);

private:
   JSONNode const &_node;
   std::set<std::string> _read;
   bool _readAll = false;
};

} // namespace Detail
} // namespace RooFit

#endif
