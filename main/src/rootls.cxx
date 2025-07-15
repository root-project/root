// \file rootls.cxx
///
/// Native implementation of rootls, partially based on rootls.py.
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

#include <TBranch.h>
#include <TError.h>
#include <TFile.h>
#include <TKey.h>
#include <TTree.h>

#include <ROOT/StringUtils.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleReader.hxx>

#if defined(R__UNIX)
#include <sys/ioctl.h>
#include <unistd.h>
#elif defined(R__WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>
#undef GetClassName
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
  -R, --rntupleListing  Print RNTuples recursively and use a long listing format.
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
   TKey *fKey = nullptr; // This is non-null for all nodes except the root node (which is the file itself)

   TDirectory *fDir = nullptr; // This is null for all non-directory nodes
   // NOTE: by construction of the tree, all children of the same node are contiguous.
   NodeIdx fFirstChild = 0;
   std::uint32_t fNChildren = 0;
   std::uint32_t fNesting = 0;
   NodeIdx fParent = 0;
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
   RootLsTree fObjectTree;
};

struct RootLsArgs {
   enum Flags {
      kNone = 0x0,
      kOneColumn = 0x1,
      kLongListing = 0x2,
      kTreeListing = 0x4,
      kRNTupleListing = 0x8,
      kRecursiveListing = 0x10,
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
   int columns = 80, rows = 25;
#if defined(R__UNIX)
   {
      winsize w;
      if (::ioctl(STDIN_FILENO, TIOCGWINSZ, &w) == 0 || ::ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 ||
          ::ioctl(STDERR_FILENO, TIOCGWINSZ, &w) == 0) {
         rows = w.ws_row;
         columns = w.ws_col;
      }
   }
#elif defined(R__WINDOWS)
   {
      CONSOLE_SCREEN_BUFFER_INFO csbi;
      if (::GetConsoleScreenBufferInfo(::GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
         columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
         rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
      }
   }
#endif
   return {columns, rows};
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
   stream << std::setw(2) << datime.GetHour() << ':' << datime.GetMinute() << ' ' << datime.GetYear() << ' ';
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
      PrintTTree(stream, *branch, indent + 2);
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

// Prints an RNTuple field tree recursively.
static void PrintRNTuple(std::ostream &stream, const ROOT::RNTupleDescriptor &desc, Indent indent,
                         const ROOT::RFieldDescriptor &rootField, std::size_t minNameLen = 0,
                         std::size_t minTypeLen = 0)
{
   std::size_t maxNameLen = 0, maxTypeLen = 0;
   std::vector<const ROOT::RFieldDescriptor *> fields;
   fields.reserve(rootField.GetLinkIds().size());
   for (const auto &field: desc.GetFieldIterable(rootField.GetId())) {
      fields.push_back(&field);
      maxNameLen = std::max(maxNameLen, field.GetFieldName().length());
      maxTypeLen = std::max(maxTypeLen, field.GetTypeName().length());
   }
   maxNameLen = std::max(minNameLen, maxNameLen + 2);
   maxTypeLen = std::max(minTypeLen, maxTypeLen + 4);

   std::sort(fields.begin(), fields.end(),
             [](const auto &a, const auto &b) { return a->GetFieldName() < b->GetFieldName(); });

   // To aid readability a bit, we use a '.' fill for all nested subfields.
   const char fillChar = minNameLen == 0 ? ' ' : '.';
   for (const auto *field : fields) {
      PrintIndent(stream, indent);
      stream << std::left << std::setfill(fillChar) << std::setw(maxNameLen) << field->GetFieldName();
      stream << std::setfill(' ') << std::setw(maxTypeLen) << field->GetTypeName();
      if (!field->GetFieldDescription().empty()) {
         std::string descStr = '"' + field->GetFieldDescription() + '"';
         stream << std::setw(1) << descStr;
      }
      stream << '\n';
      PrintRNTuple(stream, desc, indent + 2, *field, maxNameLen - 2, maxTypeLen - 2);
   }
}

static void PrintChildrenDetailed(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags,
                                  Indent indent, std::size_t minNameLen = 0, std::size_t minClassLen = 0);

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
/// \param nodesBegin The first node to be printed
/// \param nodesEnd The last node to be printed
/// \param flags A bitmask of RootLsArgs::Flags that influence how stuff is printed
/// \param indent Each line of the output will have these many leading whitespaces
static void PrintNodesDetailed(std::ostream &stream, const RootLsTree &tree,
                               std::vector<NodeIdx>::const_iterator nodesBegin,
                               std::vector<NodeIdx>::const_iterator nodesEnd, std::uint32_t flags, Indent indent,
                               std::size_t minNameLen = 0, std::size_t minClassLen = 0)
{
   std::size_t maxClassLen = 0, maxNameLen = 0;
   for (auto childIt = nodesBegin; childIt != nodesEnd; ++childIt) {
      const auto &child = tree.fNodes[*childIt];
      maxClassLen = std::max(maxClassLen, child.fClassName.length());
      maxNameLen = std::max(maxNameLen, child.fName.length());
   }
   maxClassLen = std::max(minClassLen, maxClassLen + 2);
   maxNameLen = std::max(minNameLen, maxNameLen + 2);

   for (auto childIt = nodesBegin; childIt != nodesEnd; ++childIt) {
      NodeIdx childIdx = *childIt;
      const auto &child = tree.fNodes[childIdx];

      const char *cycleStr = "";
      // If this key is the first one in the list, or if it has a different name than the previous one, it means that
      // it's the first object of that kind in the list.
      if (childIt == nodesBegin || child.fName != tree.fNodes[childIdx - 1].fName) {
         // Then we check the following key. If the current key is not the last key in the list and if the following key
         // has the same name, then it means it's another cycle of the same object. Thus, it's gonna be a backup cycle
         // of the same object. Otherwise, it's just a key with one cycle so we don't need to print information two
         // distinguish between different cycles of the same key.
         if (std::next(childIt) != nodesEnd && child.fName == tree.fNodes[childIdx + 1].fName) {
            cycleStr = "[current cycle]";
         }
      } else {
         // This key is a subsequent cycle of a previous key
         cycleStr = "[backup cycle]";
      }

      PrintIndent(stream, indent);
      stream << std::left;
      stream << Color(kAnsiBold) << std::setw(maxClassLen) << child.fClassName << Color(kAnsiNone);
      PrintDatime(stream, child.fKey->GetDatime());
      std::string namecycle = child.fName + ';' + std::to_string(child.fKey->GetCycle());
      stream << std::left << std::setw(maxNameLen) << namecycle;
      stream << " \"" << child.fKey->GetTitle() << "\" " << cycleStr;
      stream << '\n';

      if (flags & RootLsArgs::kTreeListing) {
         if (ClassInheritsFrom(child.fClassName.c_str(), "TTree")) {
            TTree *ttree = child.fKey->ReadObject<TTree>();
            if (ttree) {
               PrintTTree(stream, *ttree, indent + 2);
               PrintClusters(stream, *ttree, indent + 2);
            }
         }
      }
      if (flags & RootLsArgs::kRNTupleListing) {
         if (ClassInheritsFrom(child.fClassName.c_str(), "ROOT::RNTuple")) {
            auto *rntuple = child.fKey->ReadObject<ROOT::RNTuple>();
            if (rntuple) {
               auto reader = ROOT::RNTupleReader::Open(*rntuple);
               const auto &desc = reader->GetDescriptor();
               PrintRNTuple(stream, desc, indent + 2, desc.GetFieldZero());
            } else {
               R__LOG_ERROR(RootLsChannel()) << "failed to read RNTuple object: " << child.fName;
            }
         }
      }
      if ((flags & RootLsArgs::kRecursiveListing) && ClassInheritsFrom(child.fClassName.c_str(), "TDirectory")) {
         PrintChildrenDetailed(stream, tree, childIdx, flags, indent + 2, maxNameLen - 2, maxClassLen - 2);
      }
   }
   stream << std::flush;
}

/// \param nodeIdx The index of the node whose children should be printed
static void PrintChildrenDetailed(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags,
                                  Indent indent, std::size_t minNameLen, std::size_t minClassLen)
{

   const auto &node = tree.fNodes[nodeIdx];
   if (node.fNChildren == 0)
      return;

   std::vector<NodeIdx> children(node.fNChildren);
   std::iota(children.begin(), children.end(), node.fFirstChild);
   PrintNodesDetailed(stream, tree, children.begin(), children.end(), flags, indent, minNameLen, minClassLen);
}

// Prints all children of `nodeIdx`-th node in a ls-like fashion.
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

   //// Do the actual printing

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
         PrintChildrenInColumns(stream, tree, childIdx, flags, indent + 2);
         mustIndent = true;
      }
   }
}

// Prints all children of `nodeIdx`-th node in a ls-like fashion.
static void PrintChildrenInColumns(std::ostream &stream, const RootLsTree &tree, NodeIdx nodeIdx, std::uint32_t flags,
                                   Indent indent)
{
   const auto &node = tree.fNodes[nodeIdx];
   if (node.fNChildren == 0)
      return;

   std::vector<NodeIdx> children(node.fNChildren);
   std::iota(children.begin(), children.end(), node.fFirstChild);
   PrintNodesInColumns(stream, tree, children.begin(), children.end(), flags, indent);
}

static std::string NodeFullPath(const RootLsTree &tree, NodeIdx nodeIdx)
{
   std::vector<const std::string *> fragments;
   const RootLsNode *node = &tree.fNodes[nodeIdx];
   NodeIdx prevParent;
   do {
      prevParent = node->fParent;
      fragments.push_back(&node->fName);
      node = &tree.fNodes[node->fParent];
   } while (node->fParent != prevParent);

   assert(!fragments.empty());

   std::string fullPath = **fragments.rbegin();
   for (auto it = std::next(fragments.rbegin()), end = fragments.rend(); it != end; ++it) {
      fullPath += '/' + **it;
   }
   return fullPath;
}

// Main entrypoint of the program
static void RootLs(const RootLsArgs &args, std::ostream &stream = std::cout)
{
   const Indent outerIndent = (args.fSources.size() > 1) * 2;
   for (const auto &source : args.fSources) {
      if (args.fSources.size() > 1) {
         stream << source.fFileName << " :\n";
      }

      if (args.fFlags & (RootLsArgs::kLongListing | RootLsArgs::kTreeListing | RootLsArgs::kRNTupleListing))
         PrintNodesDetailed(stream, source.fObjectTree, source.fObjectTree.fLeafList.begin(),
                            source.fObjectTree.fLeafList.end(), args.fFlags, outerIndent);
      else
         PrintNodesInColumns(stream, source.fObjectTree, source.fObjectTree.fLeafList.begin(),
                             source.fObjectTree.fLeafList.end(), args.fFlags, outerIndent);

      const bool manySources = source.fObjectTree.fDirList.size() + source.fObjectTree.fLeafList.size() > 1;
      const Indent indent = outerIndent + manySources * 2;
      for (NodeIdx rootIdx : source.fObjectTree.fDirList) {
         if (manySources) {
            PrintIndent(stream, outerIndent);
            stream << NodeFullPath(source.fObjectTree, rootIdx) << " :\n";
         }

         if (args.fFlags & (RootLsArgs::kLongListing | RootLsArgs::kTreeListing | RootLsArgs::kRNTupleListing))
            PrintChildrenDetailed(stream, source.fObjectTree, rootIdx, args.fFlags, indent);
         else
            PrintChildrenInColumns(stream, source.fObjectTree, rootIdx, args.fFlags, indent);
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
   nodeTree.fFile = std::unique_ptr<TFile>(TFile::Open(std::string(fileName).c_str(), "READ"));
   if (!nodeTree.fFile)
      return nodeTree;

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

   // Keep track of the object names found at every nesting level and only add the first one.
   std::unordered_set<std::string> namesFound;

   const bool isRecursive = flags & RootLsArgs::kRecursiveListing;
   do {
      NodeIdx curIdx = nodesToVisit.front();
      nodesToVisit.pop_front();
      RootLsNode *cur = &nodeTree.fNodes[curIdx];
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

         if (ClassInheritsFrom(key->GetClassName(), "TDirectory"))
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
                           MatchLongFlag(arg, "recursiveListing", RootLsArgs::kRecursiveListing, outArgs.fFlags) ||
                           MatchLongFlag(arg, "rntupleListing", RootLsArgs::kRNTupleListing, outArgs.fFlags);
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
                              MatchShortFlag(*arg, 'r', RootLsArgs::kRecursiveListing, outArgs.fFlags) ||
                              MatchShortFlag(*arg, 'R', RootLsArgs::kRNTupleListing, outArgs.fFlags);
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

      // Handle known URI prefixes
      static const char *const specialPrefixes[] = {"http", "https", "root", "gs", "s3"};
      for (const char *prefix : specialPrefixes) {
         const auto prefixLen = strlen(prefix);
         if (strncmp(arg, prefix, prefixLen) == 0 && strncmp(arg + prefixLen, "://", 3) == 0) {
            newSource.fFileName = std::string(prefix) + "://";
            arg += prefixLen + 3;
            break;
         }
      }

      auto tokens = ROOT::Split(arg, ":");
      if (tokens.empty())
         continue;

      newSource.fFileName += tokens[0];
      if (tokens.size() > 1) {
         newSource.fObjectTree = GetMatchingPathsInFile(newSource.fFileName, tokens[1], outArgs.fFlags);
      } else {
         newSource.fObjectTree = GetMatchingPathsInFile(newSource.fFileName, "", outArgs.fFlags);
      }
   }

   return outArgs;
}

int main(int argc, char **argv)
{
   // Ignore diagnostics up to (but excluding) kError to avoid spamming users with TClass::Init warnings.
   gErrorIgnoreLevel = kError;

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
