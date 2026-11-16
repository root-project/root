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

#ifndef RooFit_Detail_WorkspaceStreamingContext_h
#define RooFit_Detail_WorkspaceStreamingContext_h

#include <memory>
#include <string>
#include <utility>
#include <vector>

class RooWorkspace;
class TFile;

namespace RooFit {
namespace Detail {

/// State shared by the objects that are streamed as part of one RooWorkspace, for as long as the workspace is
/// written or read. Classes with large payloads use it to store them outside of the workspace buffer, which is
/// limited to 1 GB. RooWorkspace::Write() creates the context for writing, and RooWorkspace::Streamer() for reading.
class WorkspaceStreamingContext {
public:
   /// State that one class keeps while the workspace is streamed.
   class Extension {
   public:
      virtual ~Extension() = default;
      /// Called after all objects of the workspace were streamed. When writing, the workspace is then already in the
      /// file.
      virtual void finalize() {}
   };

   WorkspaceStreamingContext(RooWorkspace const &workspace, TFile *file);
   ~WorkspaceStreamingContext();

   WorkspaceStreamingContext(WorkspaceStreamingContext const &) = delete;
   WorkspaceStreamingContext &operator=(WorkspaceStreamingContext const &) = delete;

   /// The context of the workspace that this thread is streaming, or nullptr.
   static WorkspaceStreamingContext *current();

   RooWorkspace const &workspace() const { return _workspace; }
   /// The file that the workspace is written to, if objects can write their payloads to it, else nullptr.
   TFile *file() const { return _file; }

   Extension *extension(std::string const &name) const;
   Extension &addExtension(std::string const &name, std::unique_ptr<Extension> extension);

   void finalize();

private:
   RooWorkspace const &_workspace;
   TFile *_file = nullptr;
   WorkspaceStreamingContext *_previous = nullptr;
   std::vector<std::pair<std::string, std::unique_ptr<Extension>>> _extensions;
};

} // namespace Detail
} // namespace RooFit

#endif
