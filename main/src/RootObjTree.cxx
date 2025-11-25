// \file RootObjTree.cxx
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-10-14

#include "RootObjTree.hxx"

#include "wildcards.hpp"

#include <TFile.h>

#include <ROOT/StringUtils.hxx>

#include <algorithm>
#include <deque>
#include <iostream>

static bool MatchesGlob(std::string_view haystack, std::string_view pattern)
{
   return wildcards::match(haystack, pattern);
}

ROOT::CmdLine::RootSource
ROOT::CmdLine::GetMatchingPathsInFile(std::string_view fileName, std::string_view pattern, std::uint32_t flags)
{
   ROOT::CmdLine::RootSource source;
   source.fFileName = fileName;
   auto &nodeTree = source.fObjectTree;
   nodeTree.fFile = std::unique_ptr<TFile>(TFile::Open(std::string(fileName).c_str(), "READ"));
   if (!nodeTree.fFile) {
      source.fErrors.push_back("Failed to open file");
      return source;
   }

   const auto patternSplits = pattern.empty() ? std::vector<std::string>{} : ROOT::Split(pattern, "/");

   // Match all objects at all nesting levels down to the deepest nesting level of `pattern` (or all nesting levels
   // if we have the "recursive listing" flag). The nodes are visited breadth-first.
   {
      ROOT::CmdLine::RootObjNode rootNode = {};
      rootNode.fName = std::string(fileName);
      rootNode.fClassName = nodeTree.fFile->Class()->GetName();
      rootNode.fDir = nodeTree.fFile.get();
      nodeTree.fNodes.emplace_back(std::move(rootNode));
   }
   std::deque<NodeIdx_t> nodesToVisit{0};

   // Keep track of the object names found at every nesting level and only add the first one.
   std::unordered_set<std::string> namesFound;

   const bool isRecursive = flags & EGetMatchingPathsFlags::kRecursive;
   do {
      NodeIdx_t curIdx = nodesToVisit.front();
      nodesToVisit.pop_front();
      ROOT::CmdLine::RootObjNode *cur = &nodeTree.fNodes[curIdx];
      assert(cur->fDir);

      // Sort the keys by name
      std::vector<TKey *> keys;
      keys.reserve(cur->fDir->GetListOfKeys()->GetEntries());
      for (TKey *key : ROOT::Detail::TRangeStaticCast<TKey>(cur->fDir->GetListOfKeys()))
         keys.push_back(key);

      std::sort(keys.begin(), keys.end(),
                [](const auto *a, const auto *b) { return strcmp(a->GetName(), b->GetName()) < 0; });

      namesFound.clear();

      for (TKey *key : keys) {
         // Don't recurse lower than requested by `pattern` unless we explicitly have the `recursive listing` flag.
         if (cur->fNesting < patternSplits.size() && !MatchesGlob(key->GetName(), patternSplits[cur->fNesting]))
            continue;

         if (namesFound.count(key->GetName()) > 0) {
            std::cerr << "WARNING: Several versions of '" << key->GetName() << "' are present in '" << fileName
                      << "'. Only the most recent will be considered.\n";
            continue;
         }
         namesFound.insert(key->GetName());

         auto &newChild = nodeTree.fNodes.emplace_back(NodeFromKey(*key));
         // Need to get back cur since the emplace_back() may have moved it.
         cur = &nodeTree.fNodes[curIdx];
         newChild.fNesting = cur->fNesting + 1;
         newChild.fParent = curIdx;
         if (!cur->fNChildren)
            cur->fFirstChild = nodeTree.fNodes.size() - 1;
         cur->fNChildren++;

         const auto *cl = TClass::GetClass(key->GetClassName());
         if (cl && cl->InheritsFrom("TDirectory"))
            newChild.fDir = cur->fDir->GetDirectory(key->GetName());
      }

      // Only recurse into subdirectories that are up to the deepest level we ask for through `pattern`.
      if (cur->fNesting < patternSplits.size() || isRecursive) {
         for (auto childIdx = cur->fFirstChild; childIdx < cur->fFirstChild + cur->fNChildren; ++childIdx) {
            auto &child = nodeTree.fNodes[childIdx];
            if (child.fDir)
               nodesToVisit.push_back(childIdx);
            else if (cur->fNesting < patternSplits.size())
               nodeTree.fLeafList.push_back(childIdx);
         }
      }
      if (cur->fNesting == patternSplits.size()) {
         if (cur->fDir)
            nodeTree.fDirList.push_back(curIdx);
         else
            nodeTree.fLeafList.push_back(curIdx);
      }
   } while (!nodesToVisit.empty());

   return source;
}

ROOT::CmdLine::RootSource ROOT::CmdLine::ParseRootSource(std::string_view sourceRaw, std::uint32_t flags)
{
   ROOT::CmdLine::RootSource source;
   const char *str = sourceRaw.data();

   // Handle known URI prefixes
   static const char *const specialPrefixes[] = {"http", "https", "root", "gs", "s3"};
   for (const char *prefix : specialPrefixes) {
      const auto prefixLen = strlen(prefix);
      if (strncmp(str, prefix, prefixLen) == 0 && strncmp(str + prefixLen, "://", 3) == 0) {
         source.fFileName = std::string(prefix) + "://";
         str += prefixLen + 3;
         break;
      }
   }

   auto tokens = ROOT::Split(str, ":");
   if (tokens.empty())
      return source;

   source.fFileName += tokens[0];
   if (tokens.size() > 1) {
      source = ROOT::CmdLine::GetMatchingPathsInFile(source.fFileName, tokens[1], flags);
   } else {
      source = ROOT::CmdLine::GetMatchingPathsInFile(source.fFileName, "", flags);
   }

   return source;
}

std::vector<ROOT::CmdLine::RootSource>
ROOT::CmdLine::ParseRootSources(const std::vector<std::string> &sourcesRaw, std::uint32_t flags)
{
   std::vector<ROOT::CmdLine::RootSource> sources;
   sources.reserve(sourcesRaw.size());

   for (const auto &srcRaw : sourcesRaw) {
      sources.push_back(ParseRootSource(srcRaw, flags));
   }

   return sources;
}

std::string ROOT::CmdLine::NodeFullPath(const ROOT::CmdLine::RootObjTree &tree, ROOT::CmdLine::NodeIdx_t nodeIdx,
                                        ROOT::CmdLine::ENodeFullPathOpt opt)
{
   const RootObjNode *node = &tree.fNodes[nodeIdx];
   std::string fullPath = node->fName;
   while (node->fParent != 0) {
      node = &tree.fNodes[node->fParent];
      fullPath = node->fName + (fullPath.empty() ? "" : "/") + fullPath;
   }
   if (opt == ENodeFullPathOpt::kIncludeFilename && nodeIdx > 0)
      fullPath = tree.fNodes[0].fName + ":" + fullPath;
   return fullPath;
}
