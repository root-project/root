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
///
/// ### Flag grouping
/// By default, if all the short flags you pass (those starting with a single `-`) are 1 character long, the parser will
/// accept grouped flags like "-abc" as equivalent to "-a -b -c". The last flag in the group may also accept an
/// argument, in which case "-abc foo" will count as "-a -b -c foo" where "foo" is the argument to "-c". If you want to
/// disable flag grouping, use:
///
/// ~~~{.cpp}
/// ROOT::RCmdLineOpts opts({ EFlagTreatment::kSimple });
/// ~~~
///
/// ### Repeated flags
/// Multiple repeated flags, like `-vvv`, are supported but must explicitly be marked as such on a per-flag basis:
/// ~~~{.cpp}
/// opts.AddFlag({"-v"}, RCmdLineOpts::EFlagType::kSwitch, "", RCmdLineOpts::kFlagAllowMultiple);
/// ~~~
/// This works both for switches and flags with arguments. `GetSwitch` returns the number of times a specific flag
/// appeared; for flags with arguments `GetFlagValues` and `GetFlagValuesAs<T>` can be used to access the values as
/// vectors.
///
/// ### Positional argument separator
/// The string "--" is treated as the positional argument separator: all strings after it will be treated as positional
/// arguments even if they start with "-".
///
/// ## Prefix flags (aka no space between flag and argument)
/// If you need your flags to support the syntax "-fXYZ" where "-f" is your flag and "XYZ" its argument, you can enable
/// that per-flag by using `RCmdLineOpts::kFlagPrefixArg`:
///
/// ~~~{.cpp}
/// opts.AddFlag({"-I", "--include"}, RCmdLineOpts::EFlagType::kWithArg, RCmdLineOpts::kFlagPrefixArg });
/// ~~~
///
/// (see EFlagTreatment for more details). This will **disable** flag grouping globally, but allows the parser to
/// interpret flags and arguments that are not separated by spaces.
/// Note that this only makes sense for flags with arguments.
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-10-09

#ifndef ROOT_OptParse
#define ROOT_OptParse

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstring>
#include <cstdint>
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
   enum class EFlagTreatment {
      /// Will result to kGrouped if you don't define any short flag longer than 1 character, otherwise kSimple.
      kDefault,
      /// `-abc` will always be treated as the single flag "-abc"
      kSimple,
      /// `-abc` will be treated as "-a -b -c". This is only valid for short flags.
      /// With this setting you cannot define short flags that are more than 1 character long nor ones that are marked
      /// with kFlagPrefixArg.
      kGrouped,
   };

   struct RSettings {
      /// Affects how flags are parsed (\see EFlagTreatment).
      EFlagTreatment fFlagTreatment;
   };

   enum class EFlagType {
      kSwitch,
      kWithArg
   };

   struct RFlag {
      std::string fName;
      std::string fValue;
      std::string fHelp;
   };

   // Technically these are bit flags, but EFlagFlag is confusing, so let's call them opts.
   enum EFlagOpt {
      /// Flag is allowed to appear multiple times (default: it's an error to see the same flag twice)
      kFlagAllowMultiple = 1 << 0,
      /// Flag argument can appear right after this flag without a space or equal sign in between.
      /// Note that marking any short flag with this disables flag grouping!
      kFlagPrefixArg = 1 << 1,
   };

private:
   RSettings fSettings;
   std::vector<RFlag> fFlags;
   std::vector<std::string> fArgs;

   struct RExpectedFlag {
      EFlagType fFlagType = EFlagType::kSwitch;
      std::string fName;
      std::string fHelp;
      // If >= 0, this flag is an alias of the RExpectedFlag at index fAlias.
      int fAlias = -1;
      bool fShort = false;
      std::uint32_t fOpts = 0;

      std::string AsStr() const { return std::string(fShort ? "-" : "--") + fName; }
   };
   std::vector<RExpectedFlag> fExpectedFlags;
   std::vector<std::string> fErrors;

   const RExpectedFlag *GetExpectedFlag(std::string_view name) const
   {
      const auto StartsWith = [](std::string_view string, std::string_view prefix) {
         return string.size() >= prefix.size() && string.substr(0, prefix.size()) == prefix;
      };

      for (const auto &flag : fExpectedFlags) {
         if (flag.fOpts & kFlagPrefixArg) {
            if (StartsWith(name, flag.fName)) {
               // NOTE: we can't have ambiguities here because we make sure that no flags share a common prefix in
               // AddFlag().
               return &flag;
            }
         } else if (flag.fName == name) {
            return &flag;
         }
      }
      return nullptr;
   }

public:
   explicit RCmdLineOpts(RSettings settings = {EFlagTreatment::kDefault}) : fSettings(settings) {}

   /// Returns all parsing errors
   const std::vector<std::string> &GetErrors() const { return fErrors; }
   /// Retrieves all positional arguments
   const std::vector<std::string> &GetArgs() const { return fArgs; }
   /// Retrieves all parsed flags
   const std::vector<RFlag> &GetFlags() const { return fFlags; }

   /// Conveniency method to print any errors to `stream`.
   /// \return true if any error was printed
   bool ReportErrors(std::ostream &stream = std::cerr) const
   {
      for (const auto &err : fErrors)
         stream << err << "\n";
      return !fErrors.empty();
   }

   /// Defines a new flag (either a switch or a flag with argument).
   /// The flag may be referred to as any of the values inside `aliases` (e.g. { "-h", "--help" }).
   /// You must pass at least 1 string inside `aliases`.
   /// All strings inside `aliases` must start with `-` or `--` and be at least 1 character long (aside the dashes).
   /// Flags starting with a single `-` are considered "short", regardless of their actual length.
   /// If all short flags are 1 character long, they may be collapsed into one and parsed as individual flags
   /// (meaning a string like "-fgk" will be parsed as "-f -g -k") and the final flag may have a following argument.
   /// This does NOT happen if any short flag is longer than 1 character, to avoid ambiguity.
   ///
   /// \param aliases All the equivalent names of this flag
   /// \param type What kind of flag is this: with arguments or not
   /// \param help Help string for the flag
   /// \param flagOpts Bitmask of EFlagOpt for additional options
   void AddFlag(std::initializer_list<std::string_view> aliases, EFlagType type = EFlagType::kSwitch,
                std::string_view help = "", std::uint32_t flagOpts = 0)
   {
      const auto IsPrefixOf = [](std::string_view a, std::string_view b) {
         return a.size() < b.size() && std::equal(a.begin(), a.end(), b.begin());
      };

      if (aliases.size() == 0)
         throw std::invalid_argument("AddFlag must receive at least 1 name for the flag!");

      if ((flagOpts & kFlagPrefixArg) && type != EFlagType::kWithArg)
         throw std::invalid_argument("Flag `" + std::string(*aliases.begin()) +
                                     "` has option kFlagPrefixArg but it's a Switch, so the option makes no sense.");

      int aliasIdx = -1;
      for (auto f : aliases) {
         auto prefixLen = f.find_first_not_of('-');
         if (prefixLen != 1 && prefixLen != 2)
            throw std::invalid_argument(std::string("Invalid flag `") + std::string(f) +
                                        "`: flags must start with '-' or '--'");
         if (f.size() == prefixLen)
            throw std::invalid_argument("Flag name cannot be empty");

         auto flagName = f.substr(prefixLen);

         // Check that we're not introducing ambiguities with prefix flags. While we're at it, also check that none
         // of the given aliases were already added.
         for (const auto &expFlag : fExpectedFlags) {
            // NOTE: we're checking against the full string, not just the flag name, to allow cases like:
            // AddFlag({"-foo", "--foo"}).
            if (expFlag.AsStr() == f)
               throw std::invalid_argument("Flag `" + expFlag.AsStr() + "` was added multiple times.");

            if (!(flagOpts & kFlagPrefixArg) && !(expFlag.fOpts & kFlagPrefixArg))
               continue;

            // At least one of expFlag and f is a prefix flag: this means that they must not share a common prefix.
            if (((expFlag.fOpts & kFlagPrefixArg) && IsPrefixOf(expFlag.fName, flagName)) ||
                ((flagOpts & kFlagPrefixArg) && IsPrefixOf(flagName, expFlag.fName))) {
               throw std::invalid_argument("Flags `" + expFlag.AsStr() + "` and `" + std::string(f) +
                                           "` have a common prefix. This causes ambiguity because at least one of them "
                                           "is marked with kFlagPrefixArg.");
            }
         }

         bool disallowsGrouping = (prefixLen == 1 && (f.size() > 2 || (flagOpts & kFlagPrefixArg)));
         if (disallowsGrouping) {
            if (fSettings.fFlagTreatment == EFlagTreatment::kDefault) {
               fSettings.fFlagTreatment = EFlagTreatment::kSimple;
            } else if (fSettings.fFlagTreatment == EFlagTreatment::kGrouped) {
               throw std::invalid_argument(
                  std::string("Flags starting with a single dash must be 1 character long when `FlagTreatment == "
                              "EFlagTreatment::kGrouped'! Cannot accept given flag `") +
                  std::string(f) + "`");
            }
         }

         RExpectedFlag expected;
         expected.fFlagType = type;
         expected.fName = flagName;
         expected.fHelp = help;
         expected.fAlias = aliasIdx;
         expected.fShort = prefixLen == 1;
         expected.fOpts = flagOpts;
         fExpectedFlags.push_back(expected);
         if (aliasIdx < 0)
            aliasIdx = fExpectedFlags.size() - 1;
      }
   }

   /// If `name` refers to a previously-defined switch (i.e. a boolean flag), returns how many times
   /// the flag appeared (this will never be more than 1 unless the flag is allowed to appear multiple times).
   /// \throws std::invalid_argument if the flag was undefined or defined as a flag with arguments
   int GetSwitch(std::string_view name) const
   {
      const auto *exp = GetExpectedFlag(name);
      if (!exp)
         throw std::invalid_argument(std::string("Flag `") + std::string(name) + "` is not expected");
      if (exp->fFlagType != EFlagType::kSwitch)
         throw std::invalid_argument(std::string("Flag `") + std::string(name) + "` is not a switch");

      std::string_view lookedUpName = name;
      if (exp->fAlias >= 0)
         lookedUpName = fExpectedFlags[exp->fAlias].fName;

      int n = 0;
      for (const auto &f : fFlags) {
         n += f.fName == lookedUpName;
      }
      return n;
   }

   /// If `name` refers to a previously-defined non-switch flag, gets its value.
   /// \throws std::invalid_argument if the flag was undefined or defined as a switch flag
   std::string_view GetFlagValue(std::string_view name) const
   {
      auto values = GetFlagValues(name);
      return values.empty() ? "" : values[0];
   }

   /// If `name` refers to a previously-defined non-switch flag, gets its values.
   /// This will never return more than 1 value unless the flag is allowed to appear multiple times.
   /// \throws std::invalid_argument if the flag was undefined or defined as a switch flag
   std::vector<std::string_view> GetFlagValues(std::string_view name) const
   {
      const auto *exp = GetExpectedFlag(name);
      if (!exp)
         throw std::invalid_argument(std::string("Flag `") + std::string(name) + "` is not expected");
      if (exp->fFlagType != EFlagType::kWithArg)
         throw std::invalid_argument(std::string("Flag `") + std::string(name) + "` is a switch, use GetSwitch()");

      std::string_view lookedUpName = name;
      if (exp->fAlias >= 0)
         lookedUpName = fExpectedFlags[exp->fAlias].fName;

      std::vector<std::string_view> values;
      for (const auto &f : fFlags) {
         if (f.fName == lookedUpName)
            values.push_back(f.fValue);
      }
      return values;
   }

   // Tries to retrieve the flag value as a type T.
   // The only supported types are integral and floating point types.
   // \return A value of type T if the flag is present and convertible
   // \return nullopt if the flag is not there
   // \throws std::invalid_argument if the flag is there but not convertible.
   template <typename T>
   std::optional<T> GetFlagValueAs(std::string_view name) const
   {
      auto values = GetFlagValuesAs<T>(name);
      return values.empty() ? std::nullopt : std::make_optional(values[0]);
   }

   // Tries to retrieve the flag values as a type T.
   // The only supported types are integral and floating point types.
   // \return An array of values of type T if the flag is present and all its values are convertible to T
   // \throws std::invalid_argument if the flag has values but any of them are not convertible.
   template <typename T>
   std::vector<T> GetFlagValuesAs(std::string_view name) const
   {
      static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);

      std::vector<T> values;
      for (auto val : GetFlagValues(name)) {
         // NOTE: on paper std::from_chars is supported since C++17, however some compilers don't properly support it
         // (e.g. it's not available at all on MacOS < 26 and only the integer overload is available in AlmaLinux 8).
         // There is also no compiler define that we can use to determine the availability, so we just use it only
         // from C++20 and hope for the best.
#if __cplusplus >= 202002L && !defined(__APPLE__)
         T converted;
         auto res = std::from_chars(val.data(), val.data() + val.size(), converted);
         if (res.ptr == val.data() + val.size() && res.ec == std::errc{}) {
            values.push_back(converted);
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

         values.push_back(converted);
#endif
      }
      return values;
   }

   void Parse(const char **args, std::size_t nArgs)
   {
      bool forcePositional = false;

      // If flag treatment is still Default by now it means we can safely group short flags (otherwise we'd have
      // already changed it to Simple).
      if (fSettings.fFlagTreatment == EFlagTreatment::kDefault)
         fSettings.fFlagTreatment = EFlagTreatment::kGrouped;

      // Contains one or more flags coming from one of the arguments (e.g. "-abc" may be split
      // into flags "a", "b", and "c", which will be stored in `argStr`).
      std::vector<std::string_view> argStr;

      for (std::size_t i = 0; i < nArgs && fErrors.empty(); ++i) {
         const char *arg = args[i];
         const char *const argOrig = arg;

         if (strcmp(arg, "--") == 0) {
            forcePositional = true;
            continue;
         }

         bool isFlag = !forcePositional && arg[0] == '-';
         if (!isFlag) {
            // positional argument
            fArgs.push_back(arg);
            continue;
         }

         ++arg;
         // Parse long or short flag and its argument into `argStr` / `nxtArgStr`.
         // Note that `argStr` may contain multiple flags in case of grouped short flags (in which case nxtArgStr
         // refers only to the last one).
         argStr.clear();
         std::string_view nxtArgStr;
         // If this is false `nxtArgStr` *must* refer to the next arg, otherwise it might or might not be.
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
            while (fSettings.fFlagTreatment == EFlagTreatment::kGrouped && argLen > 1) {
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
               fErrors.push_back(std::string("Unknown flag: ") + argOrig);
               break;
            }

            // In Prefix mode, check if the returned expected flag is shorter than `argS`. This can mean two things:
            // - if `nxtArgIsTentative == false` then this flag was followed by an equal sign, and in that case
            //   the intention is interpreted as "I want this flag's argument to be whatever follows the equal sign",
            //   which means we treat this as an unknown flag;
            // - otherwise, we use the rest of `argS` as the argument to the flag.
            // More concretely: if the user added flag "-D" and argS is "-Dfoo=bar", we parse it as
            // {flag: "-Dfoo", arg: "bar"}, rather than {flag: "-D", arg: "foo=bar"}.
            if ((exp->fOpts & kFlagPrefixArg) && argS.size() > exp->fName.size()) {
               if (nxtArgIsTentative) {
                  i -= !nxtArgStr.empty(); // if we had already picked a candidate next arg, undo that.
                  nxtArgStr = argS.substr(exp->fName.size());
                  nxtArgIsTentative = false;
               } else {
                  fErrors.push_back(std::string("Unknown flag: ") + argOrig);
                  break;
               }
            } else {
               assert(exp->fName.size() == argS.size());
            }

            std::string_view nxtArg = (j == argStr.size() - 1) ? nxtArgStr : "";

            RCmdLineOpts::RFlag flag;
            flag.fHelp = exp->fHelp;
            // If the flag is an alias (e.g. long version of a short one), save its name as the aliased one, so we
            // can fetch the value later by using any of the aliases.
            if (exp->fAlias < 0)
               flag.fName = exp->fName;
            else
               flag.fName = fExpectedFlags[exp->fAlias].fName;

            // Check for duplicate flags
            if (!(exp->fOpts & kFlagAllowMultiple)) {
               auto existingIt =
                  std::find_if(fFlags.begin(), fFlags.end(), [&flag](const auto &f) { return f.fName == flag.fName; });
               if (existingIt != fFlags.end()) {
                  std::string err = std::string("Flag ") + exp->AsStr() + " appeared more than once";
                  if (exp->fFlagType == RCmdLineOpts::EFlagType::kWithArg)
                     err += " with the value: " + existingIt->fValue;
                  fErrors.push_back(err);
                  break;
               }
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
};

} // namespace ROOT

#endif
