import unittest

import ROOT

# All values of the EColor enum
ecolor_values = [
    "kAsh",
    "kAzure",
    "kBlack",
    "kBlue",
    "kBrown",
    "kCyan",
    "kGrape",
    "kGray",
    "kGreen",
    "kMagenta",
    "kOrange",
    "kP10Ash",
    "kP10Blue",
    "kP10Brown",
    "kP10Cyan",
    "kP10Gray",
    "kP10Green",
    "kP10Orange",
    "kP10Red",
    "kP10Violet",
    "kP10Yellow",
    "kP6Blue",
    "kP6Grape",
    "kP6Gray",
    "kP6Red",
    "kP6Violet",
    "kP6Yellow",
    "kP8Azure",
    "kP8Blue",
    "kP8Cyan",
    "kP8Gray",
    "kP8Green",
    "kP8Orange",
    "kP8Pink",
    "kP8Red",
    "kPink",
    "kRed",
    "kSpring",
    "kTeal",
    "kViolet",
    "kWhite",
    "kYellow",
]


class TColorTests(unittest.TestCase):
    """
    Tests related to TColor.
    """

    # Tests
    def test_implicit_tcolor_from_string(self):
        """Test that we can set colors using Python strings."""

        lgnd = ROOT.TLegend(0.53, 0.73, 0.87, 0.87)

        for string_val in ecolor_values:
            enum_val = getattr(ROOT, string_val)
            lgnd.SetFillColor(string_val)
            # Check consistency
            assert enum_val == lgnd.GetFillColor()


if __name__ == "__main__":
    unittest.main()
