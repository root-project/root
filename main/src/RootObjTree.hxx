// \file RootObjTree.hxx
///
/// Utility functions used by command line tools to parse "path-like" strings like: "foo.root:dir/obj*" into a
/// tree structure usable to iterate the matched objects.
///
/// For example usage, see rootls.cxx
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-10-14

#ifndef ROOT_CMDLINE_OBJTREE
#define ROOT_CMDLINE_OBJTREE

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <TKey.h>

class TDirectory;
class TFile;

namespace ROOT::CmdLine {

using NodeIdx_t = std::uint32_t;

struct RootObjNode {
   std::string fName;
   std::string fClassName;
   TKey *fKey = nullptr; // This is non-null for all nodes except the root node (which is the file itself)

   TDirectory *fDir = nullptr; // This is null for all non-directory nodes
   // NOTE: by construction of the tree, all children of the same node are contiguous.
   NodeIdx_t fFirstChild = 0;
   std::uint32_t fNChildren = 0;
   std::uint32_t fNesting = 0;
   NodeIdx_t fParent = 0;
};

inline RootObjNode NodeFromKey(TKey &key)
{
   RootObjNode node = {};
   node.fName = key.GetName();
   node.fClassName = key.GetClassName();
   node.fKey = &key;
   return node;
}

struct RootObjTree {
   // 0th node is the root node
   std::vector<RootObjNode> fNodes;
   std::vector<NodeIdx_t> fDirList;
   std::vector<NodeIdx_t> fLeafList;
   // The file must be kept alive in order to access the nodes' keys
   std::unique_ptr<TFile> fFile;
};

struct RootSource {
   std::string fFileName;
   RootObjTree fObjectTree;
   std::vector<std::string> fErrors;
};

enum EGetMatchingPathsFlags {
   /// Recurse into subdirectories when matching objects
   kRecursive = 1 << 0,
};

/// Given a file and a "path pattern", returns a RootSource containing the tree of matched objects.
///
/// \param fileName The name of the ROOT file to look into
/// \param pattern A glob-like pattern (basically a `ls` pattern). May be empty to match anything.
/// \param flags A bitmask of EGetMatchingPathsFlags
RootSource GetMatchingPathsInFile(std::string_view fileName, std::string_view pattern, std::uint32_t flags);

/// Given a string like "file.root:dir/obj", converts it to a RootSource.
/// The string may start with one of the known file protocols: "http", "https", "root", "gs", "s3"
/// (e.g. "https://file.root").
///
/// If the source fails to get created, its fErrors list will be non-empty.
///
/// \param flags A bitmask of EGetMatchingPathsFlags
/// \return The converted source.
RootSource ParseRootSource(std::string_view sourceRaw, std::uint32_t flags);

/// Given a list of strings like "file.root:dir/obj", converts each string to a RootSource.
/// The string may start with one of the known file protocols: "http", "https", "root", "gs", "s3"
/// (e.g. "https://file.root").
///
/// If one or more sources fail to get created, each sources's fErrors list will be non-empty.
///
/// \param flags A bitmask of EGetMatchingPathsFlags
/// \return The list of converted sources.
std::vector<ROOT::CmdLine::RootSource>
ParseRootSources(const std::vector<std::string> &sourcesRaw, std::uint32_t flags);

} // namespace ROOT::CmdLine

#endif
