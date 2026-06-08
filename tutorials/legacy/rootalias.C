/// \file
/// \ingroup tutorial_legacy
/// Defines aliases:
///   - `ls(path)`
///   - `edit(filename)`
///   - `dir(path)`
///   - `pwd()`
///   - `cd(path)`
///
/// \macro_code
///
/// \author Rene Brun

#include <TSystem.h>
#include <TROOT.h>
#include <TCanvas.h>
#include <TPaveText.h>
#include <TText.h>

#include <string>
#include <cstdlib>   // std::getenv
#include <cstring>   // std::strcmp
#include <algorithm> // std::replace
#include <cstdio>    // std::printf

namespace {
inline bool IsWindows()
{
   return std::strcmp(gSystem->GetName(), "WinNT") == 0;
}
inline bool IsMac()
{
   return std::strcmp(gSystem->GetName(), "Macosx") == 0;
}

// Minimal, pragmatic shell quoting for filenames/paths.
// Good enough for normal paths; not a full shell-escaping library.
std::string QuoteForShell(const std::string &s)
{
   if (s.empty())
      return "''";

   if (IsWindows()) {
      // Wrap in double quotes; replace embedded " with ' (rare in paths)
      std::string q = s;
      std::replace(q.begin(), q.end(), '"', '\'');
      return "\"" + q + "\"";
   } else {
      // POSIX: single-quote, escape internal single quotes with '"'"'
      std::string out;
      out.reserve(s.size() + 2);
      out.push_back('\'');
      for (char c : s) {
         if (c == '\'')
            out += "'\"'\"'";
         else
            out.push_back(c);
      }
      out.push_back('\'');
      return out;
   }
}

std::string GetEnvOrEmpty(const char *name)
{
   if (const char *v = std::getenv(name))
      return std::string(v);
   return {};
}

// Build a command that (on POSIX) returns immediately to keep the ROOT prompt usable.
std::string MaybeBackground(std::string cmd)
{
   if (!IsWindows())
      cmd += " &";
   return cmd;
}
} // namespace

//______________________________________________________________________________
// Open a file in the user's editor, with robust cross-platform fallbacks.
void edit(const char *file)
{
   const std::string f = (file ? file : "");
   const std::string qf = QuoteForShell(f);
   const std::string editor = GetEnvOrEmpty("EDITOR");

   std::string cmd;

   if (IsWindows()) {
      // Use "start" to detach a new window (cmd.exe builtin).
      if (!editor.empty())
         cmd = "start " + editor + " " + qf;
      else
         cmd = "start notepad " + qf;
   } else if (IsMac()) {
      // macOS: prefer $EDITOR, else TextEdit
      if (!editor.empty())
         cmd = MaybeBackground(editor + " " + qf);
      else
         cmd = MaybeBackground("open -e " + qf);
   } else {
      // Linux/Unix: $EDITOR if set; else xdg-open; else xterm+vi
      if (!editor.empty())
         cmd = MaybeBackground(editor + " " + qf);
      else
         cmd =
            MaybeBackground("(command -v xdg-open >/dev/null 2>&1 && xdg-open " + qf + ") || (xterm -e vi " + qf + ")");
   }

   gSystem->Exec(cmd.c_str());
}

//______________________________________________________________________________
// List a directory in a compact, friendly way.
void ls(const char *path = nullptr)
{
   std::string cmd = IsWindows() ? "dir /w" : "ls";
   if (path && *path) {
      cmd += " ";
      cmd += QuoteForShell(path);
   }
   gSystem->Exec(cmd.c_str());
}

//______________________________________________________________________________
// More verbose directory view (traditional Unix-y default).
void dir(const char *path = nullptr)
{
   std::string cmd = IsWindows() ? "dir" : "ls -alF";
   if (path && *path) {
      cmd += " ";
      cmd += QuoteForShell(path);
   }
   gSystem->Exec(cmd.c_str());
}

//______________________________________________________________________________
// Return current working directory (keeps macro API stable: returns const char*).
const char *pwd()
{
   static std::string wd; // static so c_str() stays valid after return
   wd = gSystem->WorkingDirectory();
   return wd.c_str();
}

//______________________________________________________________________________
// Change directory; if no path is given, just report where we are.
const char *cd(const char *path = nullptr)
{
   if (path && *path)
      gSystem->ChangeDirectory(path);
   return pwd();
}

// ===
// The following benchmark helper (seen in your file) is kept as-is in spirit,
// just minor cleanups for clarity. If you have more helpers in your local copy,
// you can apply the same style: std::string, const-correctness, early returns.
// ===

TCanvas *bench = nullptr;

//______________________________________________________________________________
// Colorize a macro name in the summary before/after execution and run it.
void bexec2(const char *macro)
{
   std::printf("in bexec dir=%s\n", pwd());
   if (gROOT->IsBatch())
      std::printf("Processing benchmark: %s\n", macro);

   if (!bench) {
      // If bench isn't prepared yet, just run the macro.
      gROOT->Macro(macro);
      return;
   }

   auto *summary = dynamic_cast<TPaveText *>(bench->GetPrimitive("TPave"));
   if (summary) {
      if (auto *tmacro = summary->GetLineWith(macro))
         tmacro->SetTextColor(4);
      bench->Modified();
      bench->Update();
   }

   gROOT->Macro(macro);

   auto *summary2 = dynamic_cast<TPaveText *>(bench->GetPrimitive("TPave"));
   if (summary2) {
      if (auto *tmacro2 = summary2->GetLineWith(macro))
         tmacro2->SetTextColor(2);
      bench->Modified();
      bench->Update();
   }
}
