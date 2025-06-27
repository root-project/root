#include <algorithm>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "wildcards.hpp"

#include <TFile.h>
#include <TKey.h>

#include <ROOT/StringUtils.hxx>
#include <ROOT/RError.hxx>

#if defined(R__UNIX)
#include <sys/ioctl.h>
#include <unistd.h>
#elif defined(R__WINDOWS)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>
#endif

static const char *const ONE_HELP = "Print content in one column";
static const char *const LONG_PRINT_HELP = "Use a long listing format.";
static const char *const TREE_PRINT_HELP = "Print tree recursively and use a long listing format.";
static const char *const RECURSIVE_PRINT_HELP = "Traverse file recursively entering any TDirectory.";
static const char *const EPILOG = R"(Examples:
- rootls example.root
  Display contents of the ROOT file 'example.root'.

- rootls example.root:dir
  Display contents of the directory 'dir' from the ROOT file 'example.root'.

- rootls example.root:*
  Display contents of the ROOT file 'example.root' and his subdirectories.

- rootls file1.root file2.root
  Display contents of ROOT files 'file1.root' and 'file2.root'.

- rootls *.root
  Display contents of ROOT files whose name ends with '.root'.

- rootls -1 example.root
  Display contents of the ROOT file 'example.root' in one column.

- rootls -l example.root
  Display contents of the ROOT file 'example.root' and use a long listing format.

- rootls -t example.root
  Display contents of the ROOT file 'example.root', use a long listing format and print trees recursively.

- rootls -r example.root
  Display contents of the ROOT file 'example.root', traversing recursively any TDirectory.
)";

static ROOT::RLogChannel &RootLsChannel()
{
   static ROOT::RLogChannel sLog("ROOTLS");
   return sLog;
}

struct SplitPath {
   std::vector<std::string> fPathFragments;
};

using NodeIdx = std::uint32_t;

struct RootLsNode {
   std::string fName;
   TDirectory *fDir = nullptr; // may be null
   // TODO: this can probably be replaced by `NodeIdx firstChild; NodeIdx nChildren;` since the children
   // should always be contiguous.
   std::vector<NodeIdx> fChildren;
   std::uint32_t fNesting = 0;
};

struct RootLsTree {
   // 0th node is the root node
   std::vector<RootLsNode> fNodes;
};

struct RootLsSource {
   std::string fFileName;
   // List of object paths that match the query
   // std::vector<SplitPath> fObjectPaths;
   RootLsTree fObjectTree;
};

struct RootLsArgs {
   enum Flags {
      kNone = 0,
      kOneColumn = 1,
      kLongListing = 2,
      kTreeListing = 4,
      kRecursiveListing = 8
   };

   std::uint32_t fFlags = 0;
   std::vector<RootLsSource> fSources;
};

template <typename F>
static void VisitBreadthFirst(const RootLsTree &tree, const F &func, NodeIdx nodeIdx = 0)
{
   // TODO: make non-recursive?
   auto &root = tree.fNodes[nodeIdx];
   func(root);
   for (auto childIdx : root.fChildren) {
      VisitBreadthFirst(tree, func, childIdx);
   }
}

struct V2i {
   int x, y;
};

static V2i GetTerminalSize()
{
#if defined(R__UNIX)
   winsize w;
   ::ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
   return {w.ws_col, w.ws_row};
#elif defined(R__WINDOWS)
   int rows = 0, columns = 0;
   CONSOLE_SCREEN_BUFFER_INFO csbi;
   if (::GetConsoleScreenBufferInfo(::GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
      columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
      rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
   }
   return {columns, rows};
#endif
   // Fallback
   return {80, 10};
}

enum EPrintFlags {
   kPrintNone = 0,
   kPrintOneColumn = 1,
};

enum Indent : int;

static void PrintChildrenInColumns(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags, Indent indent)
{
   const auto &node = tree.fNodes[nodeIdx];
   if (node.fChildren.empty())
      return;

   const V2i terminalSize = GetTerminalSize();
   std::cout << "size: " << terminalSize.x << ", " << terminalSize.y << "\n";
   const int minCharsBetween = 2;
   const auto [minElemWidthIt, maxElemWidthIt] =
      std::minmax_element(node.fChildren.begin(), node.fChildren.end(), [&tree](NodeIdx aIdx, NodeIdx bIdx) {
         const auto &a = tree.fNodes[aIdx];
         const auto &b = tree.fNodes[bIdx];
         return a.fName.length() < b.fName.length();
      });
   const auto minElemWidth = tree.fNodes[*minElemWidthIt].fName.length() + minCharsBetween;
   const auto maxElemWidth = tree.fNodes[*maxElemWidthIt].fName.length() + minCharsBetween;

   // Figure out how many columns do we need
   int nCols = 0;
   std::vector<int> colWidths;
   if (maxElemWidth > terminalSize.x) {
      nCols = 1;
      colWidths = {1};
   } else {
      bool oneColumn = (flags & kPrintOneColumn);
      // Start with the max possible number of columns and reduce it until it fits
      nCols = oneColumn ? 1 : std::min<int>(node.fChildren.size(), terminalSize.x / static_cast<int>(minElemWidth));
      while (1) {
         int totWidth = 0;

         // Find maximum width of each column 
         for (int colIdx = 0; colIdx < nCols; ++colIdx) {
            int width = 0;
            for (int j = 0; j < node.fChildren.size(); ++j) {
               if ((j % nCols) == colIdx) {
                  NodeIdx childIdx = node.fChildren[j];
                  const RootLsNode &child = tree.fNodes[childIdx];
                  width = std::max<int>(width, child.fName.length()) + minCharsBetween;
               }
            }

            totWidth += width;
            if (totWidth > terminalSize.x) {
               --nCols;
               colWidths.clear();
               break;
            }

            colWidths.push_back(width);
         }

         if (!colWidths.empty())
            break;

         // The loop should always end at some point given the check on maxElemWidth <= terminalSize.x
         assert(nCols > 0);
      }
   }

   for (auto i = 0u; i < node.fChildren.size(); ++i) {
      NodeIdx childIdx = node.fChildren[i];
      const auto &child = tree.fNodes[childIdx];
      if (i % nCols) {
         std::string indentStr(indent, ' ');
         stream << indent;
      }
      // TODO: colors
      if (((i + 1) % nCols) != 0 && i != node.fChildren.size() - 1) {
         stream << std::left << std::setw(colWidths[i % nCols]) << child.fName;
      } else {
         stream << child.fName << "\n";
      }
   }
}

static void RootLs(const RootLsArgs &args)
{
   int indent = 2 * (args.fSources.size() > 1);
   for (const auto &source : args.fSources) {
      // std::cout << source.fFileName << "\n";
      // VisitBreadthFirst(source.fObjectTree, [](const auto &node) {
      //    std::string indentStr(std::size_t(node.fNesting * 2), ' ');
      //    std::cout << indentStr << node.fName << "\n";
      // });
      PrintChildrenInColumns(std::cout, source.fObjectTree, NodeIdx(0), kPrintNone, Indent(0));

      auto file = std::unique_ptr<TFile>(TFile::Open(source.fFileName.c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
      if (!file || file->IsZombie()) {
         R__LOG_ERROR(RootLsChannel()) << "failed to open file " << source.fFileName;
         continue;
      }

      // Print
      if (args.fSources.size() > 1) {
         std::cout << source.fFileName << ":\n";
      }
   }
}

static bool MatchesGlob(std::string_view haystack, std::string_view pattern)
{
   return wildcards::match(haystack, pattern);
}

static RootLsTree GetMatchingPathsInFile(std::string_view fileName, std::string_view pattern)
{
   // std::vector<SplitPath> result;

   auto file = std::unique_ptr<TFile>(TFile::Open(std::string(fileName).c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
   if (!file)
      return {};

   // @Speed: avoid allocating
   const auto patternSplits = ROOT::Split(pattern, "/");

   // Match all objects at all nesting levels
   struct DirLevel {
      TDirectory *fDir;
      // @Speed this is super inefficient!
      std::vector<std::string> fAncestorPaths;
   };

   RootLsTree nodeTree;
   auto &rootNode = nodeTree.fNodes.emplace_back(RootLsNode{std::string(fileName), file.get()});
   std::deque<NodeIdx> nodesToVisit{0};

   do {
      NodeIdx curIdx = nodesToVisit.front();
      nodesToVisit.pop_front();
      RootLsNode *cur = &nodeTree.fNodes[curIdx];
      assert(cur->fDir);
      if (cur->fNesting == patternSplits.size())
         break;

      for (TKey *key : ROOT::Detail::TRangeStaticCast<TKey>(cur->fDir->GetListOfKeys())) {
         if (!MatchesGlob(key->GetName(), patternSplits[cur->fNesting]))
            continue;

         RootLsNode &newChild = nodeTree.fNodes.emplace_back();
         // Need to get back cur since the emplace_back() may have moved it.
         cur = &nodeTree.fNodes[curIdx];
         newChild.fName = key->GetName();
         newChild.fNesting = cur->fNesting + 1;
         cur->fChildren.push_back(nodeTree.fNodes.size() - 1);

         const TClass *className = TClass::GetClass(key->GetClassName());
         if (className && className->InheritsFrom("TDirectory")) {
            newChild.fDir = cur->fDir->GetDirectory(key->GetName());
         }
      }

      for (auto childIdx : cur->fChildren) {
         auto &child = nodeTree.fNodes[childIdx];
         if (child.fDir)
            nodesToVisit.push_back(childIdx);
      }
   } while (!nodesToVisit.empty());

   // std::vector<DirLevel> dirsToVisit { {file.get(), {}} };

   // for (const auto &pat : patternSplits) {
   //   std::vector<DirLevel> newDirsToVisit;
   //   for (const auto &[dir, ancestorPaths] : dirsToVisit) {
   //     for (TKey *key : ROOT::Detail::TRangeStaticCast<TKey>(dir->GetListOfKeys())) {
   //       // std::cout << "matching pattern fragment " << pat << " against " << key->GetName() << "\n";
   //       if (!MatchesGlob(key->GetName(), pat))
   //         continue;

   //       const TClass *className = TClass::GetClass(key->GetClassName());
   //       std::vector<std::string> newAncestorPaths = ancestorPaths;
   //       newAncestorPaths.push_back(key->GetName());
   //       // std::cout << "matched. is dir? " << className->InheritsFrom("TDirectory") << " (class: " <<
   //       key->GetClassName() << ", name: " << key->GetName() << ")\n"; if (className &&
   //       className->InheritsFrom("TDirectory")) {
   //         newDirsToVisit.push_back({dir->GetDirectory(key->GetName()), newAncestorPaths});
   //       } else {
   //         result.push_back(SplitPath { newAncestorPaths });
   //       }
   //     }
   //   }
   //   std::swap(dirsToVisit, newDirsToVisit);
   // }

   return nodeTree;
}

static bool
MatchFlag(const char *flag, char short_, const char *long_, RootLsArgs::Flags flagVal, std::uint32_t &outFlags)
{
   const int flagLen = strlen(flag);
   if (flagLen == 1 && *flag == short_) {
      outFlags |= flagVal;
      return true;
   } else {
      const int longLen = strlen(long_);
      if (flagLen == longLen + 1 && flag[0] == '-' && strncmp(flag + 1, long_, flagLen) == 0) {
         outFlags |= flagVal;
         return true;
      }
   }
   return false;
}

static RootLsArgs ParseArgs(const char **args, int nArgs)
{
   RootLsArgs outArgs;

   for (int i = 0; i < nArgs; ++i) {
      const char *arg = args[i];
      if (arg[0] == '-') {
         ++arg;
         MatchFlag(arg, '1', "oneColumn", RootLsArgs::kOneColumn, outArgs.fFlags) ||
            MatchFlag(arg, 'l', "longListing", RootLsArgs::kLongListing, outArgs.fFlags) ||
            MatchFlag(arg, 't', "treeListing", RootLsArgs::kTreeListing, outArgs.fFlags) ||
            MatchFlag(arg, 'r', "recursiveListing", RootLsArgs::kRecursiveListing, outArgs.fFlags);
      } else {
         RootLsSource &newSource = outArgs.fSources.emplace_back();
         auto tokens = ROOT::Split(arg, ":");
         newSource.fFileName = tokens[0];
         if (tokens.size() > 1) {
            newSource.fObjectTree = GetMatchingPathsInFile(tokens[0], tokens[1]);
         }
      }
   }

   return outArgs;
}

int main(int argc, char **argv)
{
   auto args = ParseArgs(const_cast<const char **>(argv) + 1, argc - 1);

   // sort by name
   std::sort(args.fSources.begin(), args.fSources.end(),
             [](const auto &a, const auto &b) { return a.fFileName < b.fFileName; });

   RootLs(args);
}
