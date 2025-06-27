#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>

#include "wildcards.hpp"

#include <TFile.h>
#include <TKey.h>

#include <ROOT/StringUtils.hxx>

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


struct SplitPath {
  std::vector<std::string> fPathFragments;
};

struct RootLsSource {
  std::string fFileName;
  // List of object paths that match the query
  std::vector<SplitPath> fObjectPaths;
};

static void RootLs(const std::vector<RootLsSource> &sourceFiles)
{
  int indent = 2 * (sourceFiles.size() > 1);
  for (const auto &source : sourceFiles) {
    std::cout << source.fFileName << "\n";
    for (const auto &path : source.fObjectPaths) {
      int ind = indent + 2;
      for (const auto &frag : path.fPathFragments) {
        for (int i = 0; i < ind; ++i)
          std::cout << ' ';
        std::cout << frag << "\n";
        ind += 2;
      }
    }
  }
}

static bool MatchesGlob(std::string_view haystack, std::string_view pattern)
{
  return wildcards::match(haystack, pattern);
}

static std::vector<SplitPath> GetMatchingPathsInFile(std::string_view fileName, std::string_view pattern)
{
  std::vector<SplitPath> result;
  
  auto file = std::unique_ptr<TFile>(TFile::Open(std::string(fileName).c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
  if (!file)
    return result;

  // @Speed: avoid allocating
  const auto patternSplits = ROOT::Split(pattern, "/");

  // Match all objects at all nesting levels
  struct DirLevel {
    TDirectory *fDir;
    // @Speed this is super inefficient!
    std::vector<std::string> fAncestorPaths;
  };
  std::vector<DirLevel> dirsToVisit { {file.get(), {}} };
  for (const auto &pat : patternSplits) {
    std::vector<DirLevel> newDirsToVisit;
    for (const auto &[dir, ancestorPaths] : dirsToVisit) {
      for (TKey *key : ROOT::Detail::TRangeStaticCast<TKey>(dir->GetListOfKeys())) {
        // std::cout << "matching pattern fragment " << pat << " against " << key->GetName() << "\n";
        if (MatchesGlob(key->GetName(), pat)) {
          const TClass *className = TClass::GetClass(key->GetClassName());
          std::vector<std::string> newAncestorPaths = ancestorPaths;
          newAncestorPaths.push_back(key->GetName());
          // std::cout << "matched. is dir? " << className->InheritsFrom("TDirectory") << " (class: " << key->GetClassName() << ", name: " << key->GetName() << ")\n";
          if (className && className->InheritsFrom("TDirectory")) {
            newDirsToVisit.push_back({dir->GetDirectory(key->GetName()), newAncestorPaths});
          } else {
            result.push_back(SplitPath { newAncestorPaths });
          }
        }
      }
    }
    std::swap(dirsToVisit, newDirsToVisit);
  }

  return result;
}

static std::vector<RootLsSource> GetPositionalArguments(const char **args, int nArgs)
{
  std::vector<RootLsSource> outArgs;
  outArgs.reserve(nArgs);
  
  for (int i = 0; i < nArgs; ++i) {
    const char *arg = args[i];
    if (arg[0] == '-')
      continue;
    
    RootLsSource &newSource = outArgs.emplace_back();
    auto tokens = ROOT::Split(arg, ":"); 
    newSource.fFileName = tokens[0];   
    if (tokens.size() > 1) {
      newSource.fObjectPaths = GetMatchingPathsInFile(tokens[0], tokens[1]);
    }
  }

  return outArgs;
}

int main(int argc, char **argv)
{
  // TODO: parse flags
  auto sourceFiles = GetPositionalArguments(const_cast<const char **>(argv) + 1, argc - 1);

  // sort by name
  std::sort(sourceFiles.begin(), sourceFiles.end(), [] (const auto &a, const auto &b) {
    return a.fFileName < b.fFileName;
  });
  
  RootLs(sourceFiles);
}
