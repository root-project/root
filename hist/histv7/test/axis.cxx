#include "gtest/gtest.h"
#include "ROOT/RAxis.hxx"

using namespace ROOT::Experimental;

// FIXME: Passing this directly as an initializer list is ambiguous because
//        the compiler doesn't know if it should convert the inner const char*
//        literals to std::string_view or std::string.
std::vector<std::string_view> labels{"abc", "de", "fghi", "j", "klmno"};

// Test RAxisConfig and conversion to concrete axis types
TEST(AxisTest, Config) {
  // Equidistant
  {
    auto test = [](const RAxisConfig& cfg, std::string_view title) {
      EXPECT_EQ(cfg.GetTitle(), title);
      EXPECT_EQ(cfg.GetNBinsNoOver(), 10);
      EXPECT_EQ(cfg.GetKind(), RAxisConfig::kEquidistant);
      EXPECT_EQ(cfg.GetBinBorders().size(), 2u);
      EXPECT_EQ(cfg.GetBinBorders()[0], 1.2);
      EXPECT_EQ(cfg.GetBinBorders()[1], 3.4);
      EXPECT_EQ(cfg.GetBinLabels().size(), 0u);

      RAxisEquidistant axis = Internal::AxisConfigToType<RAxisConfig::kEquidistant>()(cfg);
      EXPECT_EQ(axis.GetTitle(), title);
      EXPECT_EQ(axis.GetNBinsNoOver(), 10);
      EXPECT_EQ(axis.GetMinimum(), 1.2);
      EXPECT_DOUBLE_EQ(axis.GetMaximum(), 3.4);
    };

    {
      SCOPED_TRACE("Equidistant axis config w/o title");
      test({10, 1.2, 3.4}, "");
    }

    {
      SCOPED_TRACE("Equidistant axis config with title");
      test({"RITLE_E", 10, 1.2, 3.4}, "RITLE_E");
    }
  }

  // Growable
  {
    auto test = [](const RAxisConfig& cfg, std::string_view title) {
      EXPECT_EQ(cfg.GetTitle(), title);
      EXPECT_EQ(cfg.GetNBinsNoOver(), 10);
      EXPECT_EQ(cfg.GetKind(), RAxisConfig::kGrow);
      EXPECT_EQ(cfg.GetBinBorders().size(), 2u);
      EXPECT_EQ(cfg.GetBinBorders()[0], 1.2);
      EXPECT_EQ(cfg.GetBinBorders()[1], 3.4);
      EXPECT_EQ(cfg.GetBinLabels().size(), 0u);

      RAxisGrow axis = Internal::AxisConfigToType<RAxisConfig::kGrow>()(cfg);
      EXPECT_EQ(axis.GetTitle(), title);
      EXPECT_EQ(axis.GetNBinsNoOver(), 10);
      EXPECT_EQ(axis.GetMinimum(), 1.2);
      EXPECT_DOUBLE_EQ(axis.GetMaximum(), 3.4);
    };

    {
      SCOPED_TRACE("Growable axis config w/o title");
      test({RAxisConfig::Grow, 10, 1.2, 3.4}, "");
    }

    {
      SCOPED_TRACE("Growable axis config with title");
      test({"RITLE_G", RAxisConfig::Grow, 10, 1.2, 3.4}, "RITLE_G");
    }
  }

  // Irregular
  {
    auto test = [](const RAxisConfig& cfg, std::string_view title) {
      EXPECT_EQ(cfg.GetTitle(), title);
      EXPECT_EQ(cfg.GetNBinsNoOver(), 3);
      EXPECT_EQ(cfg.GetKind(), RAxisConfig::kIrregular);
      EXPECT_EQ(cfg.GetBinBorders().size(), 4u);
      EXPECT_EQ(cfg.GetBinBorders()[0], 2.3);
      EXPECT_EQ(cfg.GetBinBorders()[1], 5.7);
      EXPECT_EQ(cfg.GetBinBorders()[2], 11.13);
      EXPECT_EQ(cfg.GetBinBorders()[3], 17.19);
      EXPECT_EQ(cfg.GetBinLabels().size(), 0u);

      RAxisIrregular axis = Internal::AxisConfigToType<RAxisConfig::kIrregular>()(cfg);
      EXPECT_EQ(axis.GetTitle(), title);
      EXPECT_EQ(axis.GetBinBorders().size(), 4u);
      EXPECT_EQ(axis.GetBinBorders()[0], 2.3);
      EXPECT_EQ(axis.GetBinBorders()[1], 5.7);
      EXPECT_EQ(axis.GetBinBorders()[2], 11.13);
      EXPECT_EQ(axis.GetBinBorders()[3], 17.19);
    };

    {
      SCOPED_TRACE("Irregular axis config w/o title");
      test({{2.3, 5.7, 11.13, 17.19}}, "");
    }

    {
      SCOPED_TRACE("Irregular axis config with title");
      test({"RITLE_I", {2.3, 5.7, 11.13, 17.19}}, "RITLE_I");
    }
  }

  // Labels
  {
    auto test = [](const RAxisConfig& cfg, std::string_view title) {
      EXPECT_EQ(cfg.GetTitle(), title);
      EXPECT_EQ(cfg.GetNBinsNoOver(), 5);
      EXPECT_EQ(cfg.GetKind(), RAxisConfig::kLabels);
      EXPECT_EQ(cfg.GetBinBorders().size(), 0u);
      EXPECT_EQ(cfg.GetBinLabels().size(), 5u);
      EXPECT_EQ(cfg.GetBinLabels()[0], "abc");
      EXPECT_EQ(cfg.GetBinLabels()[1], "de");
      EXPECT_EQ(cfg.GetBinLabels()[2], "fghi");
      EXPECT_EQ(cfg.GetBinLabels()[3], "j");
      EXPECT_EQ(cfg.GetBinLabels()[4], "klmno");

      RAxisLabels axis = Internal::AxisConfigToType<RAxisConfig::kLabels>()(cfg);
      EXPECT_EQ(axis.GetTitle(), title);
      EXPECT_EQ(axis.GetBinLabels().size(), 5u);
      EXPECT_EQ(axis.GetBinLabels()[0], "abc");
      EXPECT_EQ(axis.GetBinLabels()[1], "de");
      EXPECT_EQ(axis.GetBinLabels()[2], "fghi");
      EXPECT_EQ(axis.GetBinLabels()[3], "j");
      EXPECT_EQ(axis.GetBinLabels()[4], "klmno");
    };

    {
      SCOPED_TRACE("Labeled axis config w/o title");
      test({labels}, "");
    }

    {
      SCOPED_TRACE("Labeled axis config with title");
      test({"RITLE_L", labels}, "RITLE_L");
    }
  }
}

TEST(AxisTest, Iterator) {
  auto it = RAxisBase::const_iterator(42);
  EXPECT_EQ(*it, 42);

  {
    auto it2 = ++it;
    EXPECT_EQ(*it, 43);
    EXPECT_EQ(*it2, 43);
    auto it3 = --it;
    EXPECT_EQ(*it, 42);
    EXPECT_EQ(*it3, 42);
  }

  {
    auto it2 = it++;
    EXPECT_EQ(*it, 43);
    EXPECT_EQ(*it2, 42);
    auto it3 = it--;
    EXPECT_EQ(*it, 42);
    EXPECT_EQ(*it3, 43);
  }

  {
    auto it2 = (it += 7);
    EXPECT_EQ(*it, 49);
    EXPECT_EQ(*it2, 49);
    auto it3 = (it -= 7);
    EXPECT_EQ(*it, 42);
    EXPECT_EQ(*it3, 42);
  }

  {
    auto it2 = it + 7;
    EXPECT_EQ(*it, 42);
    EXPECT_EQ(*it2, 49);
    auto it3 = 7 + it;
    EXPECT_EQ(*it, 42);
    EXPECT_EQ(*it3, 49);
    auto it4 = it - 7;
    EXPECT_EQ(*it, 42);
    EXPECT_EQ(*it4, 35);
  }

  {
    auto it2 = RAxisBase::const_iterator(54);
    EXPECT_EQ(it2 - it, 12);
    EXPECT_EQ(it[12], 54);
  }

  {
    auto it_m1 = RAxisBase::const_iterator(41);
    auto it_p1 = RAxisBase::const_iterator(43);

    EXPECT_EQ(it < it_m1, false);
    EXPECT_EQ(it < it, false);
    EXPECT_EQ(it < it_p1, true);

    EXPECT_EQ(it > it_m1, true);
    EXPECT_EQ(it > it, false);
    EXPECT_EQ(it > it_p1, false);

    EXPECT_EQ(it <= it_m1, false);
    EXPECT_EQ(it <= it, true);
    EXPECT_EQ(it <= it_p1, true);

    EXPECT_EQ(it >= it_m1, true);
    EXPECT_EQ(it >= it, true);
    EXPECT_EQ(it >= it_p1, false);

    EXPECT_EQ(it == it_m1, false);
    EXPECT_EQ(it == it, true);
    EXPECT_EQ(it == it_p1, false);

    EXPECT_EQ(it != it_m1, true);
    EXPECT_EQ(it != it, false);
    EXPECT_EQ(it != it_p1, true);
  }
}

// Common test items for RAxisBase child classes
void test_axis_base(const RAxisBase& axis,
                    std::string_view title,
                    bool can_grow,
                    int n_bins_no_over) {
  EXPECT_EQ(axis.GetTitle(), title);
  EXPECT_EQ(axis.CanGrow(), can_grow);
  EXPECT_EQ(axis.GetNBinsNoOver(), n_bins_no_over);

  const int n_overflow_bins = can_grow ? 0 : 2;
  EXPECT_EQ(axis.GetNOverflowBins(), n_overflow_bins);
  EXPECT_EQ(axis.GetNBins(), n_bins_no_over + n_overflow_bins);

  const int underflow_bin = can_grow ? -1 : 0;
  EXPECT_EQ(axis.GetUnderflowBin(), underflow_bin);
  EXPECT_EQ(axis.IsUnderflowBin(underflow_bin-1), true);
  EXPECT_EQ(axis.IsUnderflowBin(underflow_bin), true);
  EXPECT_EQ(axis.IsUnderflowBin(underflow_bin+1), false);

  const int overflow_bin = underflow_bin + n_bins_no_over + 1;
  EXPECT_EQ(axis.GetOverflowBin(), overflow_bin);
  EXPECT_EQ(axis.IsOverflowBin(overflow_bin-1), false);
  EXPECT_EQ(axis.IsOverflowBin(overflow_bin), true);
  EXPECT_EQ(axis.IsOverflowBin(overflow_bin+1), true);

  EXPECT_EQ(*axis.begin(), underflow_bin+1);
  EXPECT_EQ(*axis.begin_with_underflow(), 0);
  EXPECT_EQ(*axis.end(), overflow_bin);
  EXPECT_EQ(*axis.end_with_overflow(), n_bins_no_over + n_overflow_bins);
}

// Common test items for RAxisEquidistant child classes
void test_axis_equidistant(const RAxisEquidistant& axis,
                           std::string_view title,
                           bool can_grow,
                           int n_bins_no_over,
                           double minimum,
                           double maximum) {
  test_axis_base(axis, title, can_grow, n_bins_no_over);

  EXPECT_DOUBLE_EQ(axis.GetMinimum(), minimum);
  EXPECT_DOUBLE_EQ(axis.GetMaximum(), maximum);
  const double bin_width = (maximum - minimum) / n_bins_no_over;
  EXPECT_DOUBLE_EQ(axis.GetBinWidth(), bin_width);
  EXPECT_DOUBLE_EQ(axis.GetInverseBinWidth(), 1.0/bin_width);

  const int underflow_findbin_res = can_grow ? RAxisBase::kIgnoreBin : 0;
  EXPECT_EQ(axis.FindBin(minimum-100*bin_width), underflow_findbin_res);
  EXPECT_EQ(axis.FindBin(minimum-0.01*bin_width), underflow_findbin_res);
  const int first_bin = can_grow ? 0 : 1;
  EXPECT_EQ(axis.FindBin(minimum+0.01*bin_width), first_bin);
  EXPECT_EQ(axis.FindBin(minimum+0.99*bin_width), first_bin);
  EXPECT_EQ(axis.FindBin(minimum+1.01*bin_width), first_bin+1);
  const int last_bin = first_bin + n_bins_no_over - 1;
  EXPECT_EQ(axis.FindBin(maximum-0.01*bin_width), last_bin);
  const int overflow_findbin_res = can_grow ? RAxisBase::kIgnoreBin : last_bin+1;
  EXPECT_EQ(axis.FindBin(maximum+0.01*bin_width), overflow_findbin_res);
  EXPECT_EQ(axis.FindBin(maximum+100*bin_width), overflow_findbin_res);

  // NOTE: Result of GetBinFrom on underflow bins, GetBinTo on overflow bins and
  //       GetBinCenter on either is considered unspecified for now. If we do
  //       ultimately decide to specify this behavior, please add a test here.
  if (!can_grow) {
    EXPECT_DOUBLE_EQ(axis.GetBinTo(0), minimum);
  }
  EXPECT_DOUBLE_EQ(axis.GetBinFrom(first_bin), minimum);
  EXPECT_DOUBLE_EQ(axis.GetBinCenter(first_bin), minimum+0.5*bin_width);
  EXPECT_DOUBLE_EQ(axis.GetBinTo(first_bin), minimum+bin_width);
  EXPECT_DOUBLE_EQ(axis.GetBinFrom(first_bin+1), minimum+bin_width);
  EXPECT_DOUBLE_EQ(axis.GetBinCenter(first_bin+1), minimum+1.5*bin_width);
  EXPECT_DOUBLE_EQ(axis.GetBinTo(first_bin+1), minimum+2*bin_width);
  EXPECT_DOUBLE_EQ(axis.GetBinFrom(last_bin), maximum-bin_width);
  EXPECT_DOUBLE_EQ(axis.GetBinCenter(last_bin), maximum-0.5*bin_width);
  EXPECT_DOUBLE_EQ(axis.GetBinTo(last_bin), maximum);
  if (!can_grow) {
    EXPECT_DOUBLE_EQ(axis.GetBinFrom(n_bins_no_over+1), maximum);
  }
  // FIXME: Can't test GetBinIndexForLowEdge as RAxis lib isn't linked in
}

// TODO: Deduplicate common test elements

TEST(AxisTest, Equidistant) {
  auto test = [](const RAxisEquidistant& axis, std::string_view title) {
    test_axis_equidistant(axis, title, false, 10, 1.2, 3.4);

    RAxisConfig cfg(axis);
    EXPECT_EQ(cfg.GetTitle(), title);
    EXPECT_EQ(cfg.GetNBinsNoOver(), 10);
    EXPECT_EQ(cfg.GetKind(), RAxisConfig::kEquidistant);
    EXPECT_EQ(cfg.GetBinBorders().size(), 2u);
    EXPECT_EQ(cfg.GetBinBorders()[0], 1.2);
    EXPECT_DOUBLE_EQ(cfg.GetBinBorders()[1], 3.4);
    EXPECT_EQ(cfg.GetBinLabels().size(), 0u);
  };

  {
    SCOPED_TRACE("Equidistant axis w/o title");
    test(RAxisEquidistant(10, 1.2, 3.4), "");
  }

  {
    SCOPED_TRACE("Equidistant axis with title");
    test(RAxisEquidistant("RITLE_E2", 10, 1.2, 3.4), "RITLE_E2");
  }

  // Only RAxisEquidistant currently has an equality operator defined
  RAxisEquidistant axis1("Title", 12, 3.4, 5.6);
  EXPECT_EQ(axis1, RAxisEquidistant("Ritle", 12, 3.4, 5.6));

  // Title is ignored by the equality operator
  EXPECT_EQ(axis1, RAxisEquidistant("Ritl", 12, 3.4, 5.6));

  // Everything else is taken into account by the equality operator
  EXPECT_NE(axis1, RAxisEquidistant("Ritle", 13, 3.4, 5.6));
  EXPECT_NE(axis1, RAxisEquidistant("Ritle", 12, 3.5, 5.6));
  EXPECT_NE(axis1, RAxisEquidistant("Ritle", 12, 3.4, 5.7));
}

TEST(AxisTest, Growable) {
  auto test = [](RAxisGrow& axis, std::string_view title) {
    const RAxisGrow& caxis = axis;

    test_axis_equidistant(caxis, title, true, 10, 1.2, 3.4);

    RAxisConfig cfg(caxis);
    EXPECT_EQ(cfg.GetTitle(), title);
    EXPECT_EQ(cfg.GetNBinsNoOver(), 10);
    EXPECT_EQ(cfg.GetKind(), RAxisConfig::kGrow);
    EXPECT_EQ(cfg.GetBinBorders().size(), 2u);
    EXPECT_EQ(cfg.GetBinBorders()[0], 1.2);
    EXPECT_DOUBLE_EQ(cfg.GetBinBorders()[1], 3.4);
    EXPECT_EQ(cfg.GetBinLabels().size(), 0u);

    // FIXME: Can't test RAxisGrow::Grow() as this method is not implemented
  };

  {
    SCOPED_TRACE("Growable axis w/o title");
    RAxisGrow grow1(10, 1.2, 3.4);
    test(grow1, "");
  }

  {
    SCOPED_TRACE("Growable axis with title");
    RAxisGrow grow2("RITLE_G2", 10, 1.2, 3.4);
    test(grow2, "RITLE_G2");
  }
}

TEST(AxisTest, Irregular) {
  auto test = [](const RAxisIrregular& axis, std::string_view title) {
    test_axis_base(axis, title, false, 3);

    EXPECT_EQ(axis.FindBin(-100), 0);
    EXPECT_EQ(axis.FindBin(2.29), 0);
    EXPECT_EQ(axis.FindBin(2.31), 1);
    EXPECT_EQ(axis.FindBin(5.69), 1);
    EXPECT_EQ(axis.FindBin(5.71), 2);
    EXPECT_EQ(axis.FindBin(11.1), 2);
    EXPECT_EQ(axis.FindBin(11.2), 3);
    EXPECT_EQ(axis.FindBin(17.1), 3);
    EXPECT_EQ(axis.FindBin(17.3), 4);
    EXPECT_EQ(axis.FindBin(1000), 4);
    EXPECT_DOUBLE_EQ(axis.GetBinCenter(0), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(axis.GetBinCenter(1), 4.0);
    EXPECT_DOUBLE_EQ(axis.GetBinCenter(2), 8.415);
    EXPECT_DOUBLE_EQ(axis.GetBinCenter(3), 14.16);
    EXPECT_DOUBLE_EQ(axis.GetBinCenter(4), std::numeric_limits<double>::max());
    EXPECT_DOUBLE_EQ(axis.GetBinFrom(0), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(axis.GetBinFrom(1), 2.3);
    EXPECT_DOUBLE_EQ(axis.GetBinFrom(2), 5.7);
    EXPECT_DOUBLE_EQ(axis.GetBinFrom(3), 11.13);
    EXPECT_DOUBLE_EQ(axis.GetBinFrom(4), 17.19);
    EXPECT_DOUBLE_EQ(axis.GetBinTo(0), 2.3);
    EXPECT_DOUBLE_EQ(axis.GetBinTo(1), 5.7);
    EXPECT_DOUBLE_EQ(axis.GetBinTo(2), 11.13);
    EXPECT_DOUBLE_EQ(axis.GetBinTo(3), 17.19);
    EXPECT_DOUBLE_EQ(axis.GetBinTo(4), std::numeric_limits<double>::max());
    EXPECT_EQ(axis.GetBinBorders().size(), 4u);
    EXPECT_EQ(axis.GetBinBorders()[0], 2.3);
    EXPECT_EQ(axis.GetBinBorders()[1], 5.7);
    EXPECT_EQ(axis.GetBinBorders()[2], 11.13);
    EXPECT_EQ(axis.GetBinBorders()[3], 17.19);

    RAxisConfig cfg(axis);
    EXPECT_EQ(cfg.GetTitle(), title);
    EXPECT_EQ(cfg.GetNBinsNoOver(), 3);
    EXPECT_EQ(cfg.GetKind(), RAxisConfig::kIrregular);
    EXPECT_EQ(cfg.GetBinBorders().size(), 4u);
    EXPECT_EQ(cfg.GetBinBorders()[0], 2.3);
    EXPECT_EQ(cfg.GetBinBorders()[1], 5.7);
    EXPECT_EQ(cfg.GetBinBorders()[2], 11.13);
    EXPECT_EQ(cfg.GetBinBorders()[3], 17.19);
    EXPECT_EQ(cfg.GetBinLabels().size(), 0u);
  };

  {
    SCOPED_TRACE("Irregular axis w/o title");
    test(RAxisIrregular({2.3, 5.7, 11.13, 17.19}), "");
  }

  {
    SCOPED_TRACE("Irregular axis with title");
    test(RAxisIrregular("RITLE_I2", {2.3, 5.7, 11.13, 17.19}), "RITLE_I2");
  }
}

TEST(AxisTest, Labels) {
  auto test = [](RAxisLabels& axis, std::string_view title) {
    // Checks which only require a const RAxisLabels&, can also be used to
    // assess state invariance after calling mutator methods which shouldn't
    // have mutated anything _else_ than their intended target.
    auto const_tests = [&title](const RAxisLabels& caxis,
                                const auto& expected_labels) {
      // Notice that the RAxisBase configuration is _not_ updated when new
      // labels are added. This is by design, according to the RAxisLabels docs.
      // The configuration would be updated on Grow(), but we can't test Grow()
      // right now since it isn't implemented yet...
      test_axis_equidistant(caxis, title, true, 5, 0.0, 5.0);

      EXPECT_EQ(caxis.GetBinLabels().size(), expected_labels.size());
      for (size_t i = 0; i < expected_labels.size(); ++i) {
        EXPECT_EQ(caxis.GetBinLabels()[i], expected_labels[i]);
      }

      RAxisConfig cfg(caxis);
      EXPECT_EQ(cfg.GetTitle(), title);
      EXPECT_EQ(cfg.GetNBinsNoOver(), static_cast<int>(expected_labels.size()));
      EXPECT_EQ(cfg.GetKind(), RAxisConfig::kLabels);
      EXPECT_EQ(cfg.GetBinBorders().size(), 0u);
      EXPECT_EQ(cfg.GetBinLabels().size(), expected_labels.size());
      for (size_t i = 0; i < expected_labels.size(); ++i) {
        EXPECT_EQ(cfg.GetBinLabels()[i], expected_labels[i]);
      }
    };
    const_tests(axis, labels);

    // Bin queries aren't const in general, but should effectively be when
    // querying bins which already exist.
    EXPECT_EQ(axis.GetBinIndex("abc"), 0);
    EXPECT_EQ(axis.GetBinIndex("de"), 1);
    EXPECT_EQ(axis.GetBinIndex("fghi"), 2);
    EXPECT_EQ(axis.GetBinIndex("j"), 3);
    EXPECT_EQ(axis.GetBinIndex("klmno"), 4);
    EXPECT_EQ(axis.GetBinCenter("abc"), 0.5);
    EXPECT_EQ(axis.GetBinCenter("de"), 1.5);
    EXPECT_EQ(axis.GetBinCenter("fghi"), 2.5);
    EXPECT_EQ(axis.GetBinCenter("j"), 3.5);
    EXPECT_EQ(axis.GetBinCenter("klmno"), 4.5);
    const_tests(axis, labels);

    // FIXME: Can't test RAxisGrow::Grow() as this method is not implemented

    // Now let's add some new bins
    auto new_labels = labels;
    EXPECT_EQ(axis.GetBinIndex("pq"), 5);
    new_labels.push_back("pq");
    const_tests(axis, new_labels);
    EXPECT_EQ(axis.GetBinCenter("pq"), 5.5);
    const_tests(axis, new_labels);
    EXPECT_EQ(axis.GetBinCenter("rst"), 6.5);
    new_labels.push_back("rst");
    const_tests(axis, new_labels);
    EXPECT_EQ(axis.GetBinIndex("rst"), 6);
    const_tests(axis, new_labels);
  };

  {
    SCOPED_TRACE("Labeled axis w/o title");
    RAxisLabels axis(labels);
    test(axis, "");
  }

  {
    SCOPED_TRACE("Labeled axis with title");
    RAxisLabels axis("RITLE_L2", labels);
    test(axis, "RITLE_L2");
  }
}

TEST(AxisTest, ReverseBinLimits) {
  {
    RAxisConfig cfg(10, 3.4, 1.2);
    EXPECT_EQ(cfg.GetBinBorders().size(), 2u);
    EXPECT_DOUBLE_EQ(cfg.GetBinBorders()[0], 1.2);
    EXPECT_DOUBLE_EQ(cfg.GetBinBorders()[1], 3.4);
    EXPECT_EQ(cfg.GetNBinsNoOver(), 10);

    // NOTE: This auto-reversal does _not_ happen when using the explicit
    //       RAxisEquidistant constructor, at the time of writing.
    //
    // RAxisEquidistant axis(10, 3.4, 1.2);
    // EXPECT_DOUBLE_EQ(axis.GetMinimum(), 1.2);
    // EXPECT_DOUBLE_EQ(axis.GetMaximum(), 3.4);
    // EXPECT_DOUBLE_EQ(axis.GetBinFrom(1), 1.2);
    // EXPECT_DOUBLE_EQ(axis.GetBinTo(10), 3.4);
    // EXPECT_EQ(axis.GetNBinsNoOver(), 10);
  }

  {
    RAxisConfig cfg(RAxisConfig::Grow, 10, 3.4, 1.2);
    EXPECT_EQ(cfg.GetBinBorders().size(), 2u);
    EXPECT_DOUBLE_EQ(cfg.GetBinBorders()[0], 1.2);
    EXPECT_DOUBLE_EQ(cfg.GetBinBorders()[1], 3.4);
    EXPECT_EQ(cfg.GetNBinsNoOver(), 10);

    // NOTE: This auto-reversal does _not_ happen when using the explicit
    //       RAxisGrow constructor, at the time of writing.
    //
    // RAxisGrow axis(10, 3.4, 1.2);
    // EXPECT_DOUBLE_EQ(axis.GetMinimum(), 1.2);
    // EXPECT_DOUBLE_EQ(axis.GetMaximum(), 3.4);
    // EXPECT_DOUBLE_EQ(axis.GetBinFrom(0), 1.2);
    // EXPECT_DOUBLE_EQ(axis.GetBinTo(9), 3.4);
    // EXPECT_EQ(axis.GetNBinsNoOver(), 10);
  }
}
