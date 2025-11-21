#!/usr/bin/env python3
"""
Test script to verify that TColor.DefinedColors works correctly in Jupyter-like environments.
This test verifies the fix for issue #20018.
"""
import sys


def test_tcolor_metadata_preservation():
    """
    Test that TColor.__init__ preserves metadata after pythonization.
    This ensures that introspection-heavy environments like Jupyter can
    properly inspect the method.
    """
    import ROOT
    
    # Check that __init__ has the expected attributes from functools.wraps
    assert hasattr(ROOT.TColor.__init__, '__wrapped__'), \
        "TColor.__init__ should have __wrapped__ attribute from functools.wraps"
    
    # Check that __name__ is preserved
    assert hasattr(ROOT.TColor.__init__, '__name__'), \
        "TColor.__init__ should have __name__ attribute"
    
    # Check that __doc__ is preserved
    assert hasattr(ROOT.TColor.__init__, '__doc__'), \
        "TColor.__init__ should have __doc__ attribute"
    
    print("Metadata preservation test: PASSED")
    return True

def test_tcolor_definedcolors():
    """
    Test the original issue: TColor.DefinedColors(1) should work without errors.
    """
    import ROOT
    
    try:
        # This was the problematic call in Jupyter notebooks
        result = ROOT.TColor.DefinedColors(1)
        print(f"TColor.DefinedColors(1) returned: {result}")
        print("DefinedColors test: PASSED")
        return True
    except Exception as e:
        print(f"TColor.DefinedColors(1) failed with error: {e}")
        print("DefinedColors test: FAILED")
        return False

def test_tcolor_full_workflow():
    """
    Test the full workflow from the original issue report.
    """
    import ROOT
    
    try:
        # Create a canvas
        ROOT.TColor.DefinedColors(1)
        c = ROOT.TCanvas("c", "Basic ROOT Plot", 800, 600)
        
        # Create a histogram with 100 bins from 0 to 10
        h = ROOT.TH1F("h", "Example Histogram;X axis;Entries", 100, 0, 10)
        
        # Fill histogram with random Gaussian numbers
        for _ in range(10000):
            h.Fill(ROOT.gRandom.Gaus(5, 1))
        
        # Draw the histogram
        h.Draw()
        
        # Draw canvas
        c.Draw()
        
        print("Full workflow test: PASSED")
        return True
    except Exception as e:
        print(f"Full workflow test failed with error: {e}")
        print("Full workflow test: FAILED")
        return False

def main():
    """
    Run all tests and return exit code.
    """
    print("=" * 60)
    print("Testing TColor metadata preservation fix for issue #20018")
    print("=" * 60)
    
    tests = [
        test_tcolor_metadata_preservation,
        test_tcolor_definedcolors,
        test_tcolor_full_workflow
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} raised exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
