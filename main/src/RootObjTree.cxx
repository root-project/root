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
#include <set>

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
   nodeTree.fFile =
      std::unique_ptr<TFile>(TFile::Open(std::string(fileName).c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
   if (!nodeTree.fFile || nodeTree.fFile->IsZombie()) {
      source.fErrors.push_back("Failed to open file");
      return source;
   }

   const auto patternSplits = pattern.empty() ? std::vector<std::string>{} : ROOT::Split(pattern, "/");
   std::vector<bool> patternWasMatchedAtLeastOnce(patternSplits.size());

   /// Match all objects at all nesting levels down to the deepest nesting level of `pattern` (or all nesting levels
   /// if we have the "recursive listing" flag). The nodes are visited breadth-first.

   // Initialize the nodeTree with the root node and mark it as the first node to visit.
   {
      ROOT::CmdLine::RootObjNode rootNode = {};
      rootNode.fName = std::string(fileName);
      rootNode.fClassName = nodeTree.fFile->Class()->GetName();
      rootNode.fDir = nodeTree.fFile.get();
      nodeTree.fNodes.emplace_back(std::move(rootNode));
   }
   std::deque<NodeIdx_t> nodesToVisit{0};

   const bool isRecursive = flags & EGetMatchingPathsFlags::kRecursive;
   do {
      NodeIdx_t curIdx = nodesToVisit.front();
      nodesToVisit.pop_front();
      ROOT::CmdLine::RootObjNode *cur = &nodeTree.fNodes[curIdx];
      assert(cur->fDir);

      // Gather all keys under this directory and sort them by namecycle.
      std::vector<TKey *> keys;
      keys.reserve(cur->fDir->GetListOfKeys()->GetEntries());
      for (TKey *key : ROOT::Detail::TRangeStaticCast<TKey>(cur->fDir->GetListOfKeys()))
         keys.push_back(key);

      std::sort(keys.begin(), keys.end(), [](const TKey *a, const TKey *b) {
         int cmp = strcmp(a->GetName(), b->GetName());
         // Note that we order by decreasing cycle, i.e. from most to least recent
         return (cmp != 0) ? (cmp < 0) : (a->GetCycle() > b->GetCycle());
      });

      // Iterate the keys and find matches
      for (TKey *key : keys) {
         // NOTE: cur->fNesting can only be >= patternSplits.size() if we have `isRecursive == true` (see the code near
         // the end of the outer do/while loop).
         // In that case we don't care about matching patterns anymore because we are already beyond the nesting level
         // where pattern filtering applies.
         // In all other cases, we check if the key name matches the pattern and skip it if it doesn't.
         if (cur->fNesting < patternSplits.size()) {
            if (MatchesGlob(key->GetName(), patternSplits[cur->fNesting]))
               patternWasMatchedAtLeastOnce[cur->fNesting] = true;
            else
               continue;
         }

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

      // Only recurse into subdirectories that are up to the deepest level we ask for through `pattern` (except in
      // case of recursive listing).
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

   if (!(flags & kIgnoreFailedMatches)) {
      for (auto i = 0u; i < patternSplits.size(); ++i) {
         // We don't append errors for '*' because its semantics imply "0 or more matches", so 0 matches is a valid
         // case. For any other pattern we expect at least 1 match.
         if (!patternWasMatchedAtLeastOnce[i] && !patternSplits[i].empty() && patternSplits[i] != "*") {
            std::string err = "'" + std::string(fileName) + ":" +
                              ROOT::Join("/", std::span<const std::string>{patternSplits.data(), i + 1}) +
                              "' matches no objects.";
            source.fErrors.push_back(err);
         }
      }
   }

   return source;
}

ROOT::RResult<std::pair<std::string_view, std::string_view>>
ROOT::CmdLine::SplitIntoFileNameAndPattern(std::string_view sourceRaw)
{
   auto prefixIdx = sourceRaw.find("://");
   std::string_view::size_type separatorIdx = 0;
   if (prefixIdx != std::string_view::npos) {
      bool prefixFound = false;
      // Handle known URI prefixes
      static const char *const specialPrefixes[] = {"http", "https", "root", "gs", "s3"};
      auto prefix = sourceRaw.substr(0, prefixIdx);
      for (std::string_view knownPrefix : specialPrefixes) {
         if (prefix == knownPrefix) {
            prefixFound = true;
            break;
         }
      }
      if (!prefixFound) {
         return R__FAIL("unknown file protocol");
      }
      separatorIdx = sourceRaw.substr(prefixIdx + 3).find_first_of(':');
      if (separatorIdx != std::string_view::npos)
         separatorIdx += prefixIdx + 3;
   } else {
      separatorIdx = sourceRaw.find_first_of(':');
   }

   if (separatorIdx != std::string_view::npos) {
      return {{sourceRaw.substr(0, separatorIdx), sourceRaw.substr(separatorIdx + 1)}};
   }
   return {{sourceRaw, std::string_view{}}};
}

ROOT::CmdLine::RootSource ROOT::CmdLine::ParseRootSource(std::string_view sourceRaw, std::uint32_t flags)
{
   ROOT::CmdLine::RootSource source;

   auto res = SplitIntoFileNameAndPattern(sourceRaw);
   if (!res) {
      source.fErrors.push_back(res.GetError()->GetReport());
      return source;
   }

   auto [fileName, tokens] = res.Unwrap();
   source = ROOT::CmdLine::GetMatchingPathsInFile(fileName, tokens, flags);

   assert(source.fErrors.empty() == !!source.fObjectTree.fFile);
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

void ROOT::CmdLine::PrintObjTree(const RootObjTree &tree, std::ostream &out)
{
   if (tree.fNodes.empty())
      return;

   struct RevNode {
      std::set<NodeIdx_t> fChildren;
   };
   std::vector<RevNode> revNodes;
   revNodes.resize(tree.fNodes.size());

   // Un-linearize the tree
   for (int i = (int)tree.fNodes.size() - 1; i >= 0; --i) {
      const auto *node = &tree.fNodes[i];
      NodeIdx_t childIdx = i;
      NodeIdx_t parentIdx = node->fParent;
      while (childIdx != parentIdx) {
         auto &revNodeParent = revNodes[parentIdx];
         revNodeParent.fChildren.insert(childIdx);
         node = &tree.fNodes[parentIdx];
         childIdx = parentIdx;
         parentIdx = node->fParent;
      }
   }

   // Print out the tree.
   // Vector of {nesting, nodeIdx}
   std::vector<std::pair<std::uint32_t, NodeIdx_t>> nodesToVisit = {{0, 0}};
   while (!nodesToVisit.empty()) {
      const auto [nesting, nodeIdx] = nodesToVisit.back();
      nodesToVisit.pop_back();
      const auto &cur = revNodes[nodeIdx];
      const auto &node = tree.fNodes[nodeIdx];
      for (auto i = 0u; i < 2 * nesting; ++i)
         out << ' ';
      out << node.fName << ";" << node.fCycle << " : " << node.fClassName << "\n";
      // Add the children in reverse order to preserve alphabetical order during depth-first visit.
      for (auto it = cur.fChildren.rbegin(); it != cur.fChildren.rend(); ++it) {
         nodesToVisit.push_back({nesting + 1, *it});
      }
   }
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
