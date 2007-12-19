#!/usr/bin/env ruby

require 'libRuby'
require 'test/unit'

class TestRR < Test::Unit::TestCase
  # Test that RR can correctly call methods with both pointer and reference
  # args
  def test_pointer_vs_referenc
    t1 = TLorentzVector.new( 1, 1, 1, 1)
    t2 = TLorentzVector.new( 1, 1, 1, 1)
    assert_nothing_raised{ t1.DeltaR( t2 ) }

    h1 = TH1F.new
    h2 = TH1F.new
    assert_nothing_raised{ h1.Add( h2 ) }

    h3 = TH1F.new
    assert_nothing_raised{ h3.Divide( h1, h2 ) }
  end
  # Test for graphics initialization. Adapted from 
  # tutorials/test/canvas.rb
  def test_pass_integer_arrays
    tc = TCanvas.new("tc", "canvas example", 200, 10, 700, 500)

    tc.SetFillColor(42)
    tc.SetGrid

    x = Array.new
    y = Array.new
    for i in 0..19 do
      x[i] = i*0.1
      y[i] = 10*Math.sin(x[i] + 0.2)
    end

    tg = TGraph.new(20, x, y)
    tg.SetLineColor(2)
    tg.SetLineWidth(4)
    tg.SetMarkerColor(4)
    tg.SetMarkerStyle(21)
    tg.Draw("ACP")

    tc.Update
    tc.GetFrame.SetFillColor(21)
    tc.GetFrame.SetBorderSize(12)
    tc.Modified
  end
  # Check the error handling in case of wrong arguments to as
  def test_as_arg_error
    tlv = TLorentzVector.new
    assert_raise( ArgumentError ) do
      tlv.as( 'WrongLorentzVector' )
    end
  end
  # Check the error handling in case of wrong arguments to new
  def test_new_arg_error
    assert_raise( ArgumentError ) do
      TLorentzVector.new( "This does not work" )
    end
  end
  # Check the error handling in case of wrong arguments for method
  def test_method_arg_error
    tlv = TLorentzVector.new
    assert_raise( ArgumentError ) do
      tlv.SetE( "This does not work" ) 
    end
  end
end
