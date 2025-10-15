// \file optparse.hxx
///
/// Small utility to parse cmdline options.
///
/// Usage:
/// ~~~{.cpp}
/// MyAppOpts ParseArgs(const char **args, int nArgs) {
///    ROOT::RCmdLineOpts opts;
///    // will parse '-c VAL', '--compress VAL' or '--compress=VAL'
///    opts.AddFlag({"-c", "--compress"}, RCmdLineOpts::EFlagType::kWithArg);
///    // will toggle a switch '--recreate' (no args).
///    opts.AddFlag({"--recreate"});
///    opts.AddFlag({"-o"}, RCmdLineOpts::EFlagType::kWithArg);
///
///    // NOTE: `args` should not contain the program name! It should usually be `argc + 1`.
///    // For example `main(char **argv, int argc)` might call this function as:
///    //    `ParseArgs(const_cast<const char**>(argv) + 1, argc - 1);`
///    opts.Parse(args, nArgs);
///
///    // Check for errors:
///    for (const auto &err : opts.GetErrors()) { /* print errors ... */ }
///    if (!opts.GetErrors().empty()) return {};
///
///    // Convert the parsed options from string if necessary:
///    MyAppOpts myOpts;
///    // switch (boolean flag):
///    myOpts.fRecreate = opts.GetSwitch("recreate");
///    // string flag:
///    myOpts.fOutput = opts.GetFlagValue("o");
///    // integer flag:
///    myOpts.fCompression = opts.GetFlagValueAs<int>("compress"); // (could also have used "c" instead of "compress")
///    // positional arguments:
///    myOpts.fArgs = opts.GetArgs();
///
///    return myOpts;
/// }
/// ~~~
///
/// ## Additional Notes
/// If all the short flags you pass (those starting with a single `-`) are 1 character long, the parser will accept
/// grouped flags like "-abc" as equivalent to "-a -b -c". The last flag in the group may also accept an argument, in
/// which case "-abc foo" will count as "-a -b -c foo" where "foo" is the argument to "-c".
///
/// Multiple repeated flags, like `-vvv` are not supported.
///
/// The string "--" is treated as the positional argument separator: all strings after it will be treated as positional
/// arguments even if they start with "-".
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-10-09

#ifndef ROOT_OptParse
#define ROOT_OptParse

#include <algorithm>
#include <charconv>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ROOT {

class RCmdLineOpts {
public:
   enum class EFlagType {
      kSwitch,
      kWithArg
   };

   struct RFlag {
      std::string fName;
      std::string fValue;
      std::string fHelp;
   };

private:
   std::vector<RFlag> fFlags;
   std::vector<std::string> fArgs;
   // If true, many short flags may be grouped: "-abc" == "-a -b -c".
   // This is automatically true if all short flags given are 1 character long, otherwise it's false.
   bool fAllowFlagGrouping = true;

   struct RExpectedFlag {
      EFlagType fFlagType = EFlagType::kSwitch;
      std::string fName;
      std::string fHelp;
      // If >= 0, this flag is an alias of the RExpectedFlag at index fAlias.
      int fAlias = -1;
      bool fShort = false;

      std::string AsStr() const { return std::string(fShort ? "-" : "--") + fName; }
   };
   std::vector<RExpectedFlag> fExpectedFlags;
   std::vector<std::string> fErrors;

   const RExpectedFlag *GetExpectedFlag(std::string_view name) const
   {
      for (const auto &flag : fExpectedFlags) {
         if (flag.fName == name)
            return &flag;
      }
      return nullptr;
   }

public:
   const std::vector<std::string> &GetErrors() const { return fErrors; }
   const std::vector<std::string> &GetArgs() const { return fArgs; }
   const std::vector<RFlag> &GetFlags() const { return fFlags; }

   /// Conveniency method to print any errors to `stream`.
   /// \return true if any error was printed
   bool ReportErrors(std::ostream &stream = std::cerr) const
   {
      for (const auto &err : fErrors)
         stream << err << "\n";
      return !fErrors.empty();
   }

   /// Defines a new flag (either a switch or a flag with argument). The flag may be referred to as any of the
   /// values inside `aliases` (e.g. { "-h", "--help" }). All strings inside `aliases` must start with `-` or `--`
   /// and be at least 1 character long (aside the dashes).
   /// Flags starting with a single `-` are considered "short", regardless of their actual length.
   /// If all short flags are 1 character long, they may be collapsed into one and parsed as individual flags
   /// (meaning a string like "-fgk" will be parsed as "-f -g -k") and the final flag may have a following argument.
   /// This does NOT happen if any short flag is longer than 1 character, to avoid ambiguity.
   void AddFlag(std::initializer_list<std::string_view> aliases, EFlagType type = EFlagType::kSwitch,
                std::string_view help = "")
   {
      int aliasIdx = -1;
      for (auto f : aliases) {
         auto prefixLen = f.find_first_not_of('-');
         if (prefixLen != 1 && prefixLen != 2)
            throw std::invalid_argument(std::string("Invalid flag `") + std::string(f) +
                                        "`: flags must start with '-' or '--'");
         if (f.size() == prefixLen)
            throw std::invalid_argument("Flag name cannot be empty");

         fAllowFlagGrouping = fAllowFlagGrouping && (prefixLen > 1 || f.size() == 2);

         RExpectedFlag expected;
         expected.fFlagType = type;
         expected.fName = f.substr(prefixLen);
         expected.fHelp = help;
         expected.fAlias = aliasIdx;
         expected.fShort = prefixLen == 1;
         fExpectedFlags.push_back(expected);
         if (aliasIdx < 0)
            aliasIdx = fExpectedFlags.size() - 1;
      }
   }

   /// If `name` refers to a previously-defined switch (i.e. a boolean flag), gets its value.
   /// \throws std::invalid_argument if the flag was undefined or defined as a flag with arguments
   bool GetSwitch(std::string_view name) const
   {
      const auto *exp = GetExpectedFlag(name);
      if (!exp)
         throw std::invalid_argument(std::string("Flag `") + std::string(name) + "` is not expected");
      if (exp->fFlagType != EFlagType::kSwitch)
         throw std::invalid_argument(std::string("Flag `") + std::string(name) + "` is not a switch");

      std::string_view lookedUpName = name;
      if (exp->fAlias >= 0)
         lookedUpName = fExpectedFlags[exp->fAlias].fName;

      for (const auto &f : fFlags) {
         if (f.fName == lookedUpName)
            return true;
      }
      return false;
   }

   /// If `name` refers to a previously-defined non-switch flag, gets its value.
   /// \throws std::invalid_argument if the flag was undefined or defined as a switch flag
   std::string_view GetFlagValue(std::string_view name) const
   {
      const auto *exp = GetExpectedFlag(name);
      if (!exp)
         throw std::invalid_argument(std::string("Flag `") + std::string(name) + "` is not expected");
      if (exp->fFlagType != EFlagType::kWithArg)
         throw std::invalid_argument(std::string("Flag `") + std::string(name) +
                                     "` is a switch, use GetSwitch()");

      std::string_view lookedUpName = name;
      if (exp->fAlias >= 0)
         lookedUpName = fExpectedFlags[exp->fAlias].fName;

      for (const auto &f : fFlags) {
         if (f.fName == lookedUpName)
            return f.fValue;
      }
      return "";
   }

   // Tries to retrieve the flag value as a type T.
   // The only supported types are integral and floating point types.
   // \return A value of type T if the flag is present and convertible
   // \return nullopt if the flag is not there
   // \throws std::invalid_argument if the flag is there but not convertible.
   template <typename T>
   std::optional<T> GetFlagValueAs(std::string_view name) const
   {
      static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);

      if (auto val = GetFlagValue(name); !val.empty()) {
         // NOTE: on paper std::from_chars is supported since C++17, however some compilers don't properly support it
         // (e.g. it's not available at all on MacOS < 26 and only the integer overload is available in AlmaLinux 8).
         // There is also no compiler define that we can use to determine the availability, so we just use it only
         // from C++20 and hope for the best.
#if __cplusplus >= 202002L && !defined(__APPLE__)
         T converted;
         auto res = std::from_chars(val.data(), val.data() + val.size(), converted);
         if (res.ptr == val.data() + val.size() && res.ec == std::errc{}) {
            return converted;
         } else {
            std::stringstream err;
            err << "Failed to parse flag `" << name << "` with value `" << val << "`";
            if constexpr (std::is_integral_v<T>)
               err << " as an integer.\n";
            else
               err << " as a floating point number.\n";

            if (res.ec == std::errc::result_out_of_range)
               throw std::out_of_range(err.str());
            else
               throw std::invalid_argument(err.str());
         }
#else
         std::conditional_t<std::is_integral_v<T>, long long, long double> converted;
         std::size_t unconvertedPos;
         if constexpr (std::is_integral_v<T>) {
            converted = std::stoll(std::string(val), &unconvertedPos);
         } else {
            converted = std::stold(std::string(val), &unconvertedPos);
         }

         const bool isOor = converted > std::numeric_limits<T>::max();
         if (unconvertedPos != val.size() || isOor) {
            std::stringstream err;
            err << "Failed to parse flag `" << name << "` with value `" << val << "`";
            if constexpr (std::is_integral_v<T>)
               err << " as an integer.\n";
            else
               err << " as a floating point number.\n";

            if (isOor)
               throw std::out_of_range(err.str());
            else
               throw std::invalid_argument(err.str());
         }

         return converted;
#endif
      }
      return std::nullopt;
   }

   void Parse(const char **args, std::size_t nArgs)
   {
      bool forcePositional = false;

      std::vector<std::string_view> argStr;

      for (std::size_t i = 0; i < nArgs && fErrors.empty(); ++i) {
         const char *arg = args[i];

         if (strcmp(arg, "--") == 0) {
            forcePositional = true;
            continue;
         }

         bool isFlag = !forcePositional && arg[0] == '-';
         if (!isFlag) {
            // positional argument
            fArgs.push_back(arg);
         } else {
            ++arg;
            // Parse long or short flag and its argument into `argStr` / `nxtArgStr`.
            // Note that `argStr` may contain multiple flags in case of grouped short flags (in which case nxtArgStr
            // refers only to the last one).
            argStr.clear();
            std::string_view nxtArgStr;
            bool nxtArgIsTentative = true;
            if (arg[0] == '-') {
               // long flag
               ++arg;
               const char *eq = strchr(arg, '=');
               if (eq) {
                  argStr.push_back(std::string_view(arg, eq - arg));
                  nxtArgStr = std::string_view(eq + 1);
                  nxtArgIsTentative = false;
               } else {
                  argStr.push_back(std::string_view(arg));
                  if (i < nArgs - 1 && args[i + 1][0] != '-') {
                     nxtArgStr = args[i + 1];
                     ++i;
                  }
               }
            } else {
               // short flag.
               // If flag grouping is active, all flags except the last one will have an implicitly empty argument.
               auto argLen = strlen(arg);
               while (fAllowFlagGrouping && argLen > 1) {
                  argStr.push_back(std::string_view{arg, 1});
                  ++arg, --argLen;
               }

               argStr.push_back(std::string_view(arg));
               if (i < nArgs - 1 && args[i + 1][0] != '-') {
                  nxtArgStr = args[i + 1];
                  ++i;
               }
            }

            for (auto j = 0u; j < argStr.size(); ++j) {
               std::string_view argS = argStr[j];
               const auto *exp = GetExpectedFlag(argS);
               if (!exp) {
                  fErrors.push_back(std::string("Unknown flag: ") + args[j]);
                  break;
               }

               std::string_view nxtArg = (j == argStr.size() - 1) ? nxtArgStr : "";

               RCmdLineOpts::RFlag flag;
               flag.fHelp = exp->fHelp;
               // If the flag is an alias (e.g. long version of a short one), save its name as the aliased one, so we
               // can fetch the value later by using any of the aliases.
               if (exp->fAlias < 0)
                  flag.fName = argS;
               else
                  flag.fName = fExpectedFlags[exp->fAlias].fName;

               // Check for duplicate flags
               auto existingIt =
                  std::find_if(fFlags.begin(), fFlags.end(), [&flag](const auto &f) { return f.fName == flag.fName; });
               if (existingIt != fFlags.end()) {
                  std::string err = std::string("Flag ") + exp->AsStr() + " appeared more than once";
                  if (exp->fFlagType == RCmdLineOpts::EFlagType::kWithArg)
                     err += " with the value: " + existingIt->fValue;
                  fErrors.push_back(err);
                  break;
               }

               // Check that arguments are what we expect.
               if (exp->fFlagType == RCmdLineOpts::EFlagType::kWithArg) {
                  if (!nxtArg.empty()) {
                     flag.fValue = nxtArg;
                  } else {
                     fErrors.push_back("Missing argument for flag " + exp->AsStr());
                  }
               } else {
                  if (!nxtArg.empty()) {
                     if (nxtArgIsTentative)
                        --i;
                     else
                        fErrors.push_back("Flag " + exp->AsStr() + " does not expect an argument");
                  }
               }

               if (!fErrors.empty())
                  break;

               fFlags.push_back(flag);
            }

            if (!fErrors.empty())
               break;
         }
      }
   }
};

} // namespace ROOT

#endif
