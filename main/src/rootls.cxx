/// \file rootls.cxx
///
/// Native implementation of rootls, vaguely based on rootls.py.
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-06-27

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
#elif defined(R__WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>
#endif

#if defined(R__WIN32)
static const char *const kAnsiNone = "";
static const char *const kAnsiGreen = "";
static const char *const kAnsiBlue = "";
static const char *const kAnsiBold = "";
#else
static const char *const kAnsiNone = "\x1B[0m";
static const char *const kAnsiGreen = "\x1B[32m";
static const char *const kAnsiBlue = "\x1B[34m";
static const char *const kAnsiBold = "\x1B[1m";
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
   std::string fClassName;
   TDatime fDatime;
   std::string fTitle;
   short fCycle = 0;

   TDirectory *fDir = nullptr; // may be null
   // TODO: this can probably be replaced by `NodeIdx firstChild; NodeIdx nChildren;` since the children
   // should always be contiguous.
   std::vector<NodeIdx> fChildren;
   std::uint32_t fNesting = 0;
};

static RootLsNode NodeFromKey(const TKey &key)
{
   RootLsNode node = {};
   node.fName = key.GetName();
   node.fClassName = key.GetClassName();
   node.fDatime = key.GetDatime();
   node.fTitle = key.GetTitle();
   node.fCycle = key.GetCycle();
   return node;
}

struct RootLsTree {
   // 0th node is the root node
   std::vector<RootLsNode> fNodes;
   std::vector<NodeIdx> fTopLevelNodes;
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

// template <typename F>
// static void VisitBreadthFirst(const RootLsTree &tree, const F &func, NodeIdx nodeIdx = 0)
// {
//    // TODO: make non-recursive?
//    auto &root = tree.fNodes[nodeIdx];
//    func(root);
//    for (auto childIdx : root.fChildren) {
//       VisitBreadthFirst(tree, func, childIdx);
//    }
// }

struct V2i {
   int x, y;
};

static V2i GetTerminalSize()
{
#if defined(R__UNIX)
   winsize w;
   if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) != 0)
      return {0, 0};
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

enum Indent : int;

static void TimeStrFromDatime(const TDatime &datime, std::ostream &os)
{
   static const char *kMonths[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
   int monthNo = datime.GetMonth() - 1;
   const char *month = monthNo >= 0 && monthNo < 12 ? kMonths[monthNo] : "???";
   std::ios defaultFmt(nullptr);
   os << month << ' ';
   os << std::right << std::setfill('0') << std::setw(2) << datime.GetDay() << ' ';
   os << datime.GetHour() << ':' << datime.GetMinute() << ' ' << datime.GetYear() << ' ';
   os.copyfmt(defaultFmt);
}

// Prints a `ls -l`-like output:
//
// $ rootls -l https://root.cern/files/tutorials/hsimple.root
// TProfile  Jun 30 23:59 2018 hprof;1  "Profile of pz versus px"
// TH1F      Jun 30 23:59 2018 hpx;1    "This is the px distribution"
// TH2F      Jun 30 23:59 2018 hpxpy;1  "py vs px"
// TNtuple   Jun 30 23:59 2018 ntuple;1 "Demo ntuple"
static void PrintChildrenDetailed(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags, Indent indent)
{
   const auto &node = tree.fNodes[nodeIdx];
   if (node.fChildren.empty())
      return;

   std::size_t maxClassLen = 0, maxNameLen = 0;
   for (NodeIdx childIdx : node.fChildren) {
      const auto &child = tree.fNodes[childIdx];
      maxClassLen = std::max(maxClassLen, child.fClassName.length());
      maxNameLen = std::max(maxNameLen, child.fName.length());
   }
   maxClassLen += 2;
   maxNameLen += 2;

   std::string indentStr;
   indentStr.assign(indent, ' ');

   for (NodeIdx childIdx : node.fChildren) {
      const auto &child = tree.fNodes[childIdx];
      std::string timeStr = ""; // TODO

      stream << indentStr;
      stream << std::left;
      stream << kAnsiBold << std::setw(maxClassLen) << child.fClassName << kAnsiNone;
      TimeStrFromDatime(child.fDatime, stream);
      std::string namecycle = child.fName + ';' + std::to_string(child.fCycle);
      stream << std::left << std::setw(maxNameLen) << namecycle;
      stream << " \"" << child.fTitle << "\"";
      stream << '\n';
   }
   stream << std::flush;
}

// Prints a `ls`-like output
static void PrintChildrenInColumns(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags, Indent indent)
{
   const auto &node = tree.fNodes[nodeIdx];
   if (node.fChildren.empty())
      return;

   const V2i terminalSize = GetTerminalSize();
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
   std::size_t nCols = 0;
   std::vector<int> colWidths;
   if (maxElemWidth > static_cast<std::size_t>(terminalSize.x)) {
      nCols = 1;
      colWidths = {1};
   } else {
      bool oneColumn = (flags & RootLsArgs::kOneColumn);
      // Start with the max possible number of columns and reduce it until it fits
      nCols = oneColumn ? 1 : std::min<int>(node.fChildren.size(), terminalSize.x / static_cast<int>(minElemWidth));
      while (1) {
         int totWidth = 0;

         // Find maximum width of each column
         for (auto colIdx = 0u; colIdx < nCols; ++colIdx) {
            int width = 0;
            for (auto j = 0u; j < node.fChildren.size(); ++j) {
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

   // Do the actual printing
   const bool isTerminal = terminalSize.x + terminalSize.y > 0;
  
   for (auto i = 0u; i < node.fChildren.size(); ++i) {
      NodeIdx childIdx = node.fChildren[i];
      const auto &child = tree.fNodes[childIdx];
      if (i % nCols) {
         for (int j = 0; j < indent; ++j)
            stream << ' ';
      }

      // Colors
      const auto *cl = TClass::GetClass(child.fClassName.c_str());
      const bool isDir = cl && cl->InheritsFrom("TDirectory");
      if (isTerminal) {
         if (isDir)
            stream << kAnsiBlue;
         else if (cl && cl->InheritsFrom("TTree"))
            stream << kAnsiGreen;
      }

      if (((i + 1) % nCols) != 0 && i != node.fChildren.size() - 1) {
         stream << std::left << std::setw(colWidths[i % nCols]) << child.fName;
      } else {
         stream << child.fName << "\n";
      }
      stream << kAnsiNone;

      if (isDir) {
         // TODO: print recursive
      }
   }
}

static void RootLs(const RootLsArgs &args)
{
   for (const auto &source : args.fSources) {
      const Indent indent = source.fObjectTree.fTopLevelNodes.size() > 1 ? Indent(2) : Indent(0);
      for (NodeIdx rootIdx : source.fObjectTree.fTopLevelNodes) {
         if (source.fObjectTree.fTopLevelNodes.size() > 1) {
            const auto &node = source.fObjectTree.fNodes[rootIdx];
            std::cout << node.fName << " :\n";
         }

         if (args.fFlags & (RootLsArgs::kLongListing | RootLsArgs::kTreeListing))
            PrintChildrenDetailed(std::cout, source.fObjectTree, rootIdx, args.fFlags, indent);
         else
            PrintChildrenInColumns(std::cout, source.fObjectTree, rootIdx, args.fFlags, indent);
      }
   }
}

static bool MatchesGlob(std::string_view haystack, std::string_view pattern)
{
   return wildcards::match(haystack, pattern);
}

static RootLsTree GetMatchingPathsInFile(std::string_view fileName, std::string_view pattern)
{
   auto file = std::unique_ptr<TFile>(TFile::Open(std::string(fileName).c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
   if (!file)
      return {};

   // @Speed: avoid allocating
   const auto patternSplits = pattern.empty() ? std::vector<std::string>{} : ROOT::Split(pattern, "/");

   // Match all objects at all nesting levels
   struct DirLevel {
      TDirectory *fDir;
   };

   RootLsTree nodeTree;
   {
      RootLsNode rootNode = {};
      rootNode.fName = std::string(fileName);
      rootNode.fClassName = file->Class()->GetName();
      rootNode.fDir = file.get();
      nodeTree.fNodes.emplace_back(std::move(rootNode));
   }
   std::deque<NodeIdx> nodesToVisit{0};

   do {
      NodeIdx curIdx = nodesToVisit.front();
      nodesToVisit.pop_front();
      RootLsNode *cur = &nodeTree.fNodes[curIdx];
      assert(cur->fDir);

      // Sort the keys by name
      std::vector<TKey *> keys;
      for (TKey *key : ROOT::Detail::TRangeStaticCast<TKey>(cur->fDir->GetListOfKeys()))
         keys.push_back(key);

      std::sort(keys.begin(), keys.end(), [] (const auto *a, const auto *b) {
         return strcmp(a->GetName(), b->GetName()) < 0;
      });
         
      for (TKey *key : keys) {
         if (cur->fNesting < patternSplits.size() && !MatchesGlob(key->GetName(), patternSplits[cur->fNesting]))
            continue;

         auto &newChild = nodeTree.fNodes.emplace_back(NodeFromKey(*key));
         // Need to get back cur since the emplace_back() may have moved it.
         cur = &nodeTree.fNodes[curIdx];
         newChild.fNesting = cur->fNesting + 1;
         cur->fChildren.push_back(nodeTree.fNodes.size() - 1);

         const TClass *cl = TClass::GetClass(key->GetClassName());
         if (cl && cl->InheritsFrom("TDirectory")) {
            newChild.fDir = cur->fDir->GetDirectory(key->GetName());
         }
      }

      // Only recurse into subdirectories that are at the deepest level we ask for through `pattern`.
      if (cur->fNesting < patternSplits.size()) {
         for (auto childIdx : cur->fChildren) {
            auto &child = nodeTree.fNodes[childIdx];
            if (child.fDir)
               nodesToVisit.push_back(childIdx);
         }
      } else {
         nodeTree.fTopLevelNodes.push_back(curIdx);
      }
   } while (!nodesToVisit.empty());

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
         } else {
            newSource.fObjectTree = GetMatchingPathsInFile(tokens[0], "");
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

   // // TEMP
   // for (const auto &src : args.fSources) {
   //    std::cout << src.fFileName << ", " << src.fObjectTree.fNodes.size() << "\n";
   // }

   RootLs(args);
}
