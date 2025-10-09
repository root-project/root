#include <gtest/gtest.h>
#include "../src/optparse.hxx"

#include <vector>

TEST(OptParse, OptParseNull)
{
   ROOT::RCmdLineOpts opts;
   opts.Parse(nullptr, 0);
   EXPECT_TRUE(opts.GetErrors().empty());
}

TEST(OptParse, OptParseEmpty)
{
   ROOT::RCmdLineOpts opts;

   const char *args[] = {""};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
}

TEST(OptParse, OptParseNoExpected)
{
   ROOT::RCmdLineOpts opts;

   const char *args[] = {"-h"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Unknown flag: -h"});
}

TEST(OptParse, OptParseBoolean)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-h", "--help"});
   opts.AddFlag({"-f"});
   opts.AddFlag({"-g"});

   const char *args[] = {"-h", "-f"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_TRUE(opts.GetSwitch("h"));
   EXPECT_TRUE(opts.GetSwitch("help"));
   EXPECT_THROW(opts.GetFlagValue("h"), std::invalid_argument);
   EXPECT_THROW(opts.GetSwitch("-h"), std::invalid_argument);
   EXPECT_THROW(opts.GetSwitch("--help"), std::invalid_argument);
   EXPECT_TRUE(opts.GetSwitch("f"));
   EXPECT_FALSE(opts.GetSwitch("g"));
}

TEST(OptParse, OptParseString)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"--foo", "bar"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_THROW(opts.GetSwitch("foo"), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("foo"), "bar");
}

TEST(OptParse, OptParseStringEq)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"--foo=bar"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_THROW(opts.GetSwitch("foo"), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("foo"), "bar");
}

TEST(OptParse, OptParseStringShort)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-f", "--foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-f", "bar"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_THROW(opts.GetSwitch("foo"), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("foo"), "bar");
}

TEST(OptParse, OptParseStringShortMultiChar)
{
   ROOT::RCmdLineOpts opts;
   // Short flags may have multiple characters (the "shortness" comes from the fact that it starts with a single `-`).
   // Note that it's legal to define a short and long flag with the same name.
   opts.AddFlag({"-foo", "--foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-foo", "bar"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_THROW(opts.GetSwitch("foo"), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("foo"), "bar");
}

TEST(OptParse, OptParseStringShortWrong)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-f", "--foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-f=bar"};
   opts.Parse(args, std::size(args));

   // NOTE: in this case `-f=bar` was parsed as grouped short flags
   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Missing argument for flag -f"});
   EXPECT_THROW(opts.GetSwitch("foo"), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("foo"), "");
}

TEST(OptParse, OptParseStringShortWrong2)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-foo=bar"};
   opts.Parse(args, std::size(args));

   // NOTE: in this case `-foo=bar` was parsed as a single short flag (the `=` is only valid for long flags)
   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Unknown flag: -foo=bar"});
   EXPECT_THROW(opts.GetSwitch("foo"), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("foo"), "");
}

TEST(OptParse, OptParseMissingArg)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);
   opts.AddFlag({"-b"});

   const char *args[] = {"--foo", "-b"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Missing argument for flag --foo"});
   EXPECT_EQ(opts.GetFlagValue("foo"), "");
   EXPECT_FALSE(opts.GetSwitch("b"));
}

TEST(OptParse, OptParseMissingArg2)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"--foo"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Missing argument for flag --foo"});
   EXPECT_EQ(opts.GetFlagValue("foo"), "");
}

TEST(OptParse, OptParseExtraArg)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo"});

   const char *args[] = {"--foo=bar"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Flag --foo does not expect an argument"});
   EXPECT_FALSE(opts.GetSwitch("foo"));
}

TEST(OptParse, OptParseRepeatedFlagsEqual)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo"});

   const char *args[] = {"--foo", "bar", "baz", "--foo"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Flag --foo appeared more than once"});
}

TEST(OptParse, OptParseRepeatedFlagsAliased)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo", "-f", "-foo"});

   const char *args[] = {"--foo", "bar", "-f"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Flag -f appeared more than once"});
}

TEST(OptParse, OptParseRepeatedFlagsWithArg)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo", "-f", "-foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"--foo", "bar", "-f", "baz"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Flag -f appeared more than once with the value: bar"});
}

TEST(OptParse, OptParseRepeatedFlagsShort)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo", "-f"});

   const char *args[] = {"-ff"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Flag -f appeared more than once"});
}

TEST(OptParse, OptParseGroupedFlags)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"});
   opts.AddFlag({"-c"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-abc", "d"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_TRUE(opts.GetSwitch("a"));
   EXPECT_TRUE(opts.GetSwitch("b"));
   EXPECT_THROW(opts.GetSwitch("c"), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("c"), "d");
}

TEST(OptParse, OptParseNonGroupedFlags)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"});
   opts.AddFlag({"-c"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);
   // This non-monocharacter short flags prevents flag grouping
   opts.AddFlag({"-wf"});

   const char *args[] = {"-abc", "d"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Unknown flag: -abc"});
}

TEST(OptParse, OptParseGroupedFlagsMissingArg)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"});
   opts.AddFlag({"-c"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-cab", "d"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Missing argument for flag -c"});
}

TEST(OptParse, OptParseGroupedFlagsMissingArg2)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);
   opts.AddFlag({"-c"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-abc", "d"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Missing argument for flag -b"});
}

TEST(OptParse, PositionalArgs)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-b", "c", "-a", "d"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_TRUE(opts.GetSwitch("a"));
   EXPECT_EQ(opts.GetFlagValue("b"), "c");
   EXPECT_EQ(opts.GetArgs(), std::vector<std::string>{"d"});
}

TEST(OptParse, OnlyPositionalArgs)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"a", "d"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_FALSE(opts.GetSwitch("a"));
   EXPECT_EQ(opts.GetFlagValue("b"), "");
   EXPECT_EQ(opts.GetArgs(), std::vector<std::string>({"a", "d"}));
}

TEST(OptParse, PositionalSeparator)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-a", "--", "-b", "d"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_TRUE(opts.GetSwitch("a"));
   EXPECT_EQ(opts.GetFlagValue("b"), "");
   EXPECT_EQ(opts.GetArgs(), std::vector<std::string>({"-b", "d"}));
}

TEST(OptParse, PositionalSeparator2)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"--", "-b", "-a"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_FALSE(opts.GetSwitch("a"));
   EXPECT_EQ(opts.GetFlagValue("b"), "");
   EXPECT_EQ(opts.GetArgs(), std::vector<std::string>({"-b", "-a"}));
}

TEST(OptParse, OnlyPositionalSeparator)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a"});
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"--"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_FALSE(opts.GetSwitch("a"));
   EXPECT_EQ(opts.GetFlagValue("b"), "");
   EXPECT_TRUE(opts.GetArgs().empty());
}

TEST(OptParse, PositionalSeparatorAsArg)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-b", "--"};
   opts.Parse(args, std::size(args));

   EXPECT_EQ(opts.GetErrors(), std::vector<std::string>{"Missing argument for flag -b"});
}

TEST(OptParse, ParseFlagAsInt)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-b", "42"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_EQ(opts.GetFlagValueAs<int>("b"), 42);
   EXPECT_EQ(opts.GetFlagValue("b"), "42");
   EXPECT_FLOAT_EQ(opts.GetFlagValueAs<float>("b").value(), 42.f);
}

TEST(OptParse, ParseFlagAsIntInvalid)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-b", "42a"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_THROW(opts.GetFlagValueAs<int>("b"), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("b"), "42a");
}

TEST(OptParse, ParseFlagAsIntInvalid2)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-b", "4.2"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_THROW(opts.GetFlagValueAs<int>("b"), std::invalid_argument);
}

TEST(OptParse, ParseFlagAsIntOutOfRange)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-b", "2000000"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_THROW(opts.GetFlagValueAs<std::uint8_t>("b"), std::out_of_range);
}

TEST(OptParse, ParseFlagAsFloat)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-b", ".2"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_FLOAT_EQ(opts.GetFlagValueAs<float>("b").value(), .2f);
   EXPECT_THROW(opts.GetFlagValueAs<int>("b").value(), std::invalid_argument);
   EXPECT_EQ(opts.GetFlagValue("b"), ".2");
}

TEST(OptParse, ParseFlagAsFloatInvalid)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"-b", "42as"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_THROW(opts.GetFlagValueAs<float>("b"), std::invalid_argument);
}

TEST(OptParse, ParseFlagAsNumericNotThere)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-b"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);

   const char *args[] = {"bb"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_EQ(opts.GetFlagValueAs<float>("b"), std::nullopt);
   EXPECT_EQ(opts.GetFlagValueAs<int>("b"), std::nullopt);
   EXPECT_EQ(opts.GetFlagValue("b"), "");
}

TEST(OptParse, PositionalAtFirstPlace)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--foo"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);
   opts.AddFlag({"-abc"});

   const char *args[] = {"somename", "--foo", "bar", "-abc"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_EQ(opts.GetFlagValue("foo"), "bar");
   EXPECT_EQ(opts.GetSwitch("abc"), true);
   EXPECT_EQ(opts.GetArgs(), std::vector<std::string>{"somename"});
}

TEST(OptParse, PositionalMixedWithFlags)
{
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"--noarg"});

   const char *args[] = {"somename", "--noarg", "bar"};
   opts.Parse(args, std::size(args));

   EXPECT_TRUE(opts.GetErrors().empty());
   EXPECT_EQ(opts.GetSwitch("noarg"), true);
   EXPECT_EQ(opts.GetArgs(), std::vector<std::string>({"somename", "bar"}));
}
