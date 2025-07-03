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
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "wildcards.hpp"

#include <TFile.h>
#include <TKey.h>
#include <TTree.h>
#include <THnSparse.h>

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

static const char *const kAnsiNone = "\x1B[0m";
static const char *const kAnsiGreen = "\x1B[32m";
static const char *const kAnsiBlue = "\x1B[34m";
static const char *const kAnsiBold = "\x1B[1m";

static const char *Color(const char *col)
{
#if defined(R__WIN32)
   return "";
#else
   const static bool isTerm = isatty(STDOUT_FILENO);
   if (isTerm)
      return col;
   return "";
#endif
}

static const char *const kLongHelp = R"(
Display ROOT files contents in the terminal.

positional arguments:
  FILE                  Input file

options:
  -h, --help            show this help message and exit
  -1, --oneColumn       Print content in one column
  -l, --longListing     Use a long listing format.
  -t, --treeListing     Print tree recursively and use a long listing format.
  -r, --recursiveListing
                        Traverse file recursively entering any TDirectory.

Examples:
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

static bool ClassInheritsFrom(const char *class_, const char *baseClass)
{
   const auto *cl = TClass::GetClass(class_);
   const bool inherits = cl && cl->InheritsFrom(baseClass);
   return inherits;
}

using NodeIdx = std::uint32_t;

struct RootLsNode {
   std::string fName;
   std::string fClassName;
   TKey *fKey = nullptr;

   TDirectory *fDir = nullptr; // may be null
   // TODO: this can probably be replaced by `NodeIdx firstChild; NodeIdx nChildren;` since the children
   // should always be contiguous.
   std::vector<NodeIdx> fChildren;
   std::uint32_t fNesting = 0;
};

static RootLsNode NodeFromKey(TKey &key)
{
   RootLsNode node = {};
   node.fName = key.GetName();
   node.fClassName = key.GetClassName();
   node.fKey = &key;
   return node;
}

struct RootLsTree {
   // 0th node is the root node
   std::vector<RootLsNode> fNodes;
   std::vector<NodeIdx> fDirList;
   std::vector<NodeIdx> fLeafList;
   // The file must be kept alive in order to access the nodes' keys
   std::unique_ptr<TFile> fFile;
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

   enum class PrintUsage {
      kNo,
      kShort,
      kLong
   };

   std::uint32_t fFlags = 0;
   std::vector<RootLsSource> fSources;
   PrintUsage fPrintUsageAndExit = PrintUsage::kNo;
};

struct V2i {
   int x, y;
};

static V2i GetTerminalSize()
{
#if defined(R__UNIX)
   winsize w;
   if (::ioctl(STDIN_FILENO, TIOCGWINSZ, &w) == 0 || ::ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 ||
       ::ioctl(STDERR_FILENO, TIOCGWINSZ, &w) == 0)
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
   return {80, 25};
}

using Indent = int;

static void PrintIndent(std::ostream &stream, Indent indent)
{
   for (int i = 0; i < indent; ++i) {
      stream << ' ';
   }
}

static void PrintDatime(std::ostream &stream, const TDatime &datime)
{
   static const char *kMonths[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
   int monthNo = datime.GetMonth() - 1;
   const char *month = monthNo >= 0 && monthNo < 12 ? kMonths[monthNo] : "???";
   std::ios defaultFmt(nullptr);
   stream << month << ' ';
   stream << std::right << std::setfill('0') << std::setw(2) << datime.GetDay() << ' ';
   stream << datime.GetHour() << ':' << datime.GetMinute() << ' ' << datime.GetYear() << ' ';
   stream.copyfmt(defaultFmt);
}

// NOTE: T may be a TTree or a TBranch
template <typename T>
static void PrintTTree(std::ostream &stream, T &tree, Indent indent)
{
   TObjArray *branches = tree.GetListOfBranches();
   std::size_t maxNameLen = 0, maxTitleLen = 0;
   for (int i = 0; i < branches->GetEntries(); ++i) {
      TBranch *branch = static_cast<TBranch *>((*branches)[i]);
      maxNameLen = std::max(maxNameLen, strlen(branch->GetName()));
      maxTitleLen = std::max(maxTitleLen, strlen(branch->GetTitle()));
   }
   maxNameLen += 2;
   maxTitleLen += 4;

   for (int i = 0; i < branches->GetEntries(); ++i) {
      TBranch *branch = static_cast<TBranch *>((*branches)[i]);
      PrintIndent(stream, indent);
      stream << std::left << std::setw(maxNameLen) << branch->GetName();
      std::string titleStr = std::string("\"") + branch->GetTitle() + "\"";
      stream << std::setw(maxTitleLen) << titleStr;
      stream << std::setw(1) << branch->GetTotBytes();
      stream << '\n';
      // @Recursion
      PrintTTree(stream, *branch, Indent(indent + 2));
   }
}

static void PrintClusters(std::ostream &stream, TTree &tree, Indent indent)
{
   PrintIndent(stream, indent);
   stream << Color(kAnsiBold) << "Cluster INCLUSIVE ranges:\n" << Color(kAnsiNone);

   std::size_t nTotClusters = 0;
   auto clusterIt = tree.GetClusterIterator(0);
   auto clusterStart = clusterIt();
   const auto nEntries = tree.GetEntries();
   while (clusterStart < nEntries) {
      PrintIndent(stream, indent);
      stream << " - # " << nTotClusters << ": [" << clusterStart << ", " << clusterIt.GetNextEntry() - 1 << "]\n";
      ++nTotClusters;
      clusterStart = clusterIt();
   }
   PrintIndent(stream, indent);
   stream << Color(kAnsiBold) << "The total number of clusters is " << nTotClusters << "\n";
}

/// Prints a `ls -l`-like output:
///
/// $ rootls -l https://root.cern/files/tutorials/hsimple.root
/// TProfile  Jun 30 23:59 2018 hprof;1  "Profile of pz versus px"
/// TH1F      Jun 30 23:59 2018 hpx;1    "This is the px distribution"
/// TH2F      Jun 30 23:59 2018 hpxpy;1  "py vs px"
/// TNtuple   Jun 30 23:59 2018 ntuple;1 "Demo ntuple"
///
/// \param stream The output stream to print to
/// \param tree The node tree
/// \param nodeIdx The index of the node whose children should be printed
/// \param flags A bitmask of RootLsArgs::Flags that influence how stuff is printed
/// \param indent Each line of the output will have these many leading whitespaces
static void
PrintChildrenDetailed(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags, Indent indent)
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

   for (NodeIdx childIdx : node.fChildren) {
      const auto &child = tree.fNodes[childIdx];
      std::string timeStr = ""; // TODO

      PrintIndent(stream, indent);
      stream << std::left;
      stream << Color(kAnsiBold) << std::setw(maxClassLen) << child.fClassName << Color(kAnsiNone);
      PrintDatime(stream, child.fKey->GetDatime());
      std::string namecycle = child.fName + ';' + std::to_string(child.fKey->GetCycle());
      stream << std::left << std::setw(maxNameLen) << namecycle;
      stream << " \"" << child.fKey->GetTitle() << "\"";
      stream << '\n';

      if (flags & RootLsArgs::kTreeListing) {
         if (ClassInheritsFrom(child.fClassName.c_str(), "TTree")) {
            TTree *tree = child.fKey->ReadObject<TTree>();
            if (tree) {
               PrintTTree(stream, *tree, Indent(indent + 2));
               PrintClusters(stream, *tree, Indent(indent + 2));
            }
         }
         if (ClassInheritsFrom(child.fClassName.c_str(), "THnSparse")) {
            THnSparse *hs = child.fKey->ReadObject<THnSparse>();
            if (hs)
               hs->Print("all");
         }
      }
      if ((flags & RootLsArgs::kRecursiveListing) && ClassInheritsFrom(child.fClassName.c_str(), "TDirectory")) {
         PrintChildrenDetailed(stream, tree, childIdx, flags, Indent(indent + 2));
      }
   }
   stream << std::flush;
}

static void PrintChildrenInColumns(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags,
                                   Indent indent);

// Prints a `ls`-like output
static void PrintNodesInColumns(std::ostream &stream, const RootLsTree &tree,
                                std::vector<NodeIdx>::const_iterator nodesBegin,
                                std::vector<NodeIdx>::const_iterator nodesEnd, std::uint32_t flags, Indent indent)
{
   const auto nNodes = std::distance(nodesBegin, nodesEnd);
   if (nNodes == 0)
      return;

   // Calculate the min and max column size
   V2i terminalSize = GetTerminalSize();
   terminalSize.x -= indent;
   const auto [minElemWidthIt, maxElemWidthIt] =
      std::minmax_element(nodesBegin, nodesEnd, [&tree](NodeIdx aIdx, NodeIdx bIdx) {
         const auto &a = tree.fNodes[aIdx];
         const auto &b = tree.fNodes[bIdx];
         return a.fName.length() < b.fName.length();
      });
   const int minCharsBetween = 2;
   const auto minElemWidth = tree.fNodes[*minElemWidthIt].fName.length() + minCharsBetween;
   const auto maxElemWidth = tree.fNodes[*maxElemWidthIt].fName.length() + minCharsBetween;

   // Figure out how many columns do we need
   std::size_t nCols = 0;
   std::vector<int> colWidths;
   const bool oneColumn = (flags & RootLsArgs::kOneColumn);
   if (maxElemWidth > static_cast<std::size_t>(terminalSize.x) || oneColumn) {
      nCols = 1;
      colWidths = {1};
   } else {
      // Start with the max possible number of columns and reduce it until it fits
      nCols = std::min<int>(nNodes, terminalSize.x / static_cast<int>(minElemWidth));
      while (1) {
         int totWidth = 0;

         // Find maximum width of each column
         for (auto colIdx = 0u; colIdx < nCols; ++colIdx) {
            int width = 0;
            for (auto j = 0u; j < nNodes; ++j) {
               if ((j % nCols) == colIdx) {
                  NodeIdx childIdx = nodesBegin[j];
                  const RootLsNode &child = tree.fNodes[childIdx];
                  width = std::max<int>(width, child.fName.length() + minCharsBetween);
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

   bool mustIndent = false;
   for (auto i = 0u; i < nNodes; ++i) {
      NodeIdx childIdx = nodesBegin[i];
      const auto &child = tree.fNodes[childIdx];
      if ((i % nCols) == 0 || mustIndent) {
         PrintIndent(stream, indent);
      }

      // Colors
      const bool isDir = ClassInheritsFrom(child.fClassName.c_str(), "TDirectory");
      if (isTerminal) {
         if (isDir)
            stream << Color(kAnsiBlue);
         else if (ClassInheritsFrom(child.fClassName.c_str(), "TTree"))
            stream << Color(kAnsiGreen);
      }

      const bool isExtremal = !(((i + 1) % nCols) != 0 && i != nNodes - 1);
      if (!isExtremal) {
         stream << std::left << std::setw(colWidths[i % nCols]) << child.fName;
      } else {
         stream << std::setw(1) << child.fName;
      }
      stream << Color(kAnsiNone);

      if (isExtremal)
         stream << "\n";

      if (isDir && (flags & RootLsArgs::kRecursiveListing)) {
         if (!isExtremal)
            stream << "\n";
         PrintChildrenInColumns(stream, tree, childIdx, flags, Indent(indent + 2));
         mustIndent = true;
      }
   }
}

static void PrintChildrenInColumns(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags,
                                   Indent indent)
{
   const auto &node = tree.fNodes[nodeIdx];
   if (node.fChildren.empty())
      return;

   PrintNodesInColumns(stream, tree, node.fChildren.begin(), node.fChildren.end(), flags, indent);
}

static void RootLs(const RootLsArgs &args)
{
   const Indent outerIndent = (args.fSources.size() > 1) * 2;
   for (const auto &source : args.fSources) {
      if (args.fSources.size() > 1) {
         std::cout << source.fFileName << " :\n";
      }
      const Indent indent = outerIndent + (source.fObjectTree.fDirList.size() > 1) * 2;
      PrintNodesInColumns(std::cout, source.fObjectTree, source.fObjectTree.fLeafList.begin(),
                          source.fObjectTree.fLeafList.end(), args.fFlags, indent);
      for (NodeIdx rootIdx : source.fObjectTree.fDirList) {
         if (source.fObjectTree.fDirList.size() > 1) {
            const auto &node = source.fObjectTree.fNodes[rootIdx];
            PrintIndent(std::cout, outerIndent);
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

/// Inspects `fileName` to match all children that match `pattern`. Returns a tree with all the matched nodes.
/// `flags` is a bitmask of `RootLsArgs::Flags`.
static RootLsTree GetMatchingPathsInFile(std::string_view fileName, std::string_view pattern, std::uint32_t flags)
{
   RootLsTree nodeTree;
   nodeTree.fFile =
      std::unique_ptr<TFile>(TFile::Open(std::string(fileName).c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
   if (!nodeTree.fFile)
      return nodeTree;

   // @Speed: avoid allocating
   const auto patternSplits = pattern.empty() ? std::vector<std::string>{} : ROOT::Split(pattern, "/");

   // Match all objects at all nesting levels down to the deepest nesting level of `pattern` (or all nesting levels
   // if we have the "recursive listing" flag). The nodes are visited breadth-first.
   {
      RootLsNode rootNode = {};
      rootNode.fName = std::string(fileName);
      rootNode.fClassName = nodeTree.fFile->Class()->GetName();
      rootNode.fDir = nodeTree.fFile.get();
      nodeTree.fNodes.emplace_back(std::move(rootNode));
   }
   std::deque<NodeIdx> nodesToVisit{0};

   const bool isRecursive = flags & RootLsArgs::kRecursiveListing;
   do {
      NodeIdx curIdx = nodesToVisit.front();
      nodesToVisit.pop_front();
      RootLsNode *cur = &nodeTree.fNodes[curIdx];
      assert(cur->fDir);

      // Sort the keys by name
      std::vector<TKey *> keys;
      for (TKey *key : ROOT::Detail::TRangeStaticCast<TKey>(cur->fDir->GetListOfKeys()))
         keys.push_back(key);

      std::sort(keys.begin(), keys.end(),
                [](const auto *a, const auto *b) { return strcmp(a->GetName(), b->GetName()) < 0; });

      for (TKey *key : keys) {
         const auto &pat = patternSplits[cur->fNesting];
         // Don't recurse lower than requested by `pattern` unless we explicitly have the `recursive listing` flag.
         if (cur->fNesting < patternSplits.size() && !MatchesGlob(key->GetName(), patternSplits[cur->fNesting]))
            continue;

         auto &newChild = nodeTree.fNodes.emplace_back(NodeFromKey(*key));
         // Need to get back cur since the emplace_back() may have moved it.
         cur = &nodeTree.fNodes[curIdx];
         newChild.fNesting = cur->fNesting + 1;
         cur->fChildren.push_back(nodeTree.fNodes.size() - 1);

         if (ClassInheritsFrom(key->GetClassName(), "TDirectory"))
            newChild.fDir = cur->fDir->GetDirectory(key->GetName());
      }

      // Only recurse into subdirectories that are at the deepest level we ask for through `pattern`.
      if (cur->fNesting < patternSplits.size() || isRecursive) {
         for (auto childIdx : cur->fChildren) {
            auto &child = nodeTree.fNodes[childIdx];
            if (child.fDir)
               nodesToVisit.push_back(childIdx);
            else
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

   return nodeTree;
}

static bool MatchShortFlag(char arg, char matched, RootLsArgs::Flags flagVal, std::uint32_t &outFlags)
{
   if (arg == matched) {
      outFlags |= flagVal;
      return true;
   }
   return false;
}

static bool MatchLongFlag(const char *arg, const char *matched, RootLsArgs::Flags flagVal, std::uint32_t &outFlags)
{
   if (strcmp(arg, matched) == 0) {
      outFlags |= flagVal;
      return true;
   }
   return false;
}

static RootLsArgs ParseArgs(const char **args, int nArgs)
{
   RootLsArgs outArgs;
   std::vector<int> sourceArgs;

   // First match all flags, then process positional arguments (since we need the flags to properly process them).
   for (int i = 0; i < nArgs; ++i) {
      const char *arg = args[i];
      if (arg[0] == '-') {
         ++arg;
         if (arg[0] == '-') {
            // long flag
            ++arg;
            bool matched = MatchLongFlag(arg, "oneColumn", RootLsArgs::kOneColumn, outArgs.fFlags) ||
                           MatchLongFlag(arg, "longListing", RootLsArgs::kLongListing, outArgs.fFlags) ||
                           MatchLongFlag(arg, "treeListing", RootLsArgs::kTreeListing, outArgs.fFlags) ||
                           MatchLongFlag(arg, "recursiveListing", RootLsArgs::kRecursiveListing, outArgs.fFlags);
            if (!matched) {
               if (strcmp(arg, "help") == 0) {
                  outArgs.fPrintUsageAndExit = RootLsArgs::PrintUsage::kLong;
               } else {
                  R__LOG_ERROR(RootLsChannel()) << "unrecognized argument: --" << arg << "\n";
                  if (outArgs.fPrintUsageAndExit == RootLsArgs::PrintUsage::kNo)
                     outArgs.fPrintUsageAndExit = RootLsArgs::PrintUsage::kShort;
               }
            }
         } else {
            // short flag
            while (*arg) {
               bool matched = MatchShortFlag(*arg, '1', RootLsArgs::kOneColumn, outArgs.fFlags) ||
                              MatchShortFlag(*arg, 'l', RootLsArgs::kLongListing, outArgs.fFlags) ||
                              MatchShortFlag(*arg, 't', RootLsArgs::kTreeListing, outArgs.fFlags) ||
                              MatchShortFlag(*arg, 'r', RootLsArgs::kRecursiveListing, outArgs.fFlags);
               if (!matched) {
                  if (*arg == 'h') {
                     outArgs.fPrintUsageAndExit = RootLsArgs::PrintUsage::kLong;
                  } else {
                     R__LOG_ERROR(RootLsChannel()) << "unrecognized argument: -" << *arg << "\n";
                     if (outArgs.fPrintUsageAndExit == RootLsArgs::PrintUsage::kNo)
                        outArgs.fPrintUsageAndExit = RootLsArgs::PrintUsage::kShort;
                  }
               }
               ++arg;
            }
         }
      } else {
         sourceArgs.push_back(i);
      }
   }

   // Positional arguments
   for (int argIdx : sourceArgs) {
      const char *arg = args[argIdx];
      RootLsSource &newSource = outArgs.fSources.emplace_back();
      auto tokens = ROOT::Split(arg, ":");
      newSource.fFileName = tokens[0];
      if (tokens.size() > 1) {
         newSource.fObjectTree = GetMatchingPathsInFile(tokens[0], tokens[1], outArgs.fFlags);
      } else {
         newSource.fObjectTree = GetMatchingPathsInFile(tokens[0], "", outArgs.fFlags);
      }
   }

   return outArgs;
}

int main(int argc, char **argv)
{
   auto args = ParseArgs(const_cast<const char **>(argv) + 1, argc - 1);
   if (args.fPrintUsageAndExit != RootLsArgs::PrintUsage::kNo) {
      std::cerr << "usage: rootls [-1hltr] FILE [FILE ...]\n";
      if (args.fPrintUsageAndExit == RootLsArgs::PrintUsage::kLong) {
         std::cerr << kLongHelp;
         return 0;
      }
      return 1;
   }

   // sort sources by name
   std::sort(args.fSources.begin(), args.fSources.end(),
             [](const auto &a, const auto &b) { return a.fFileName < b.fFileName; });

   // sort leaves by name
   for (auto &source : args.fSources) {
      std::sort(source.fObjectTree.fLeafList.begin(), source.fObjectTree.fLeafList.end(),
                [&tree = source.fObjectTree](NodeIdx aIdx, NodeIdx bIdx) {
                   const auto &a = tree.fNodes[aIdx];
                   const auto &b = tree.fNodes[bIdx];
                   return a.fName < b.fName;
                });
   }

   RootLs(args);
}
