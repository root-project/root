/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/Detail/WorkspaceStreamingContext.h>

#include <stdexcept>

namespace {

thread_local RooFit::Detail::WorkspaceStreamingContext *currentContext = nullptr;

} // namespace

namespace RooFit {
namespace Detail {

WorkspaceStreamingContext::WorkspaceStreamingContext(RooWorkspace const &workspace, TFile *file)
   : _workspace{workspace}, _file{file}, _previous{currentContext}
{
   currentContext = this;
}

WorkspaceStreamingContext::~WorkspaceStreamingContext()
{
   currentContext = _previous;
}

WorkspaceStreamingContext *WorkspaceStreamingContext::current()
{
   return currentContext;
}

/// The extension with a given name, or nullptr if there is none.
WorkspaceStreamingContext::Extension *WorkspaceStreamingContext::extension(std::string const &name) const
{
   for (auto const &item : _extensions) {
      if (item.first == name) {
         return item.second.get();
      }
   }
   return nullptr;
}

WorkspaceStreamingContext::Extension &
WorkspaceStreamingContext::addExtension(std::string const &name, std::unique_ptr<Extension> extension)
{
   if (this->extension(name)) {
      throw std::invalid_argument("WorkspaceStreamingContext: there is already an extension named \"" + name + "\"");
   }
   _extensions.emplace_back(name, std::move(extension));
   return *_extensions.back().second;
}

/// Finalize the extensions in the order in which they were added.
void WorkspaceStreamingContext::finalize()
{
   for (auto const &item : _extensions) {
      item.second->finalize();
   }
}

} // namespace Detail
} // namespace RooFit
