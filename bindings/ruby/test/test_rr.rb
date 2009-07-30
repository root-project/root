#!/usr/bin/env ruby

require 'libRuby'
require 'test/unit'

class DRRAbstractClass
  # Additional inspect for RR to include debugging information
  def rr_inspect
    ary = Array.new
    ary << "<RR: "
    ary << "<#class:" << self.class << ">"
    ary << "<#ClassName:" << self.ClassName << ">"
    ary << "<#Class#GetName:" << self.Class.GetName << ">"
    ary << " >"
    return ary.join
  end
end

class TestRR < Test::Unit::TestCase
  # Test that RR can correctly call methods with both pointer and reference
  # args
  def test_pointer_vs_reference
    t1 = TLorentzVector.new( 1, 1, 1, 1)
    t2 = TLorentzVector.new( 1, 1, 1, 1)
    assert_nothing_raised{ t1.DeltaR( t2 ) }

    h1 = TH1F.new
    h2 = TH1F.new
    assert_nothing_raised{ h1.Add( h2 ) }

    h3 = TH1F.new
    assert_nothing_raised{ h3.Divide( h1, h2 ) }

    # Make sure that calling Add (or any method that passes a pointer) works
    # the second time from drr_generic_method. This was broken by r21428.
    ts = THStack.new
    ts.Add( TH1F.new( rand.to_s, "1", 5, 0, 5 ) ) 
    ts.Add( TH1F.new( rand.to_s, "2", 5, 0, 5 ) ) 
    # This fails in r21428, since the new mapping is not used universally
    assert_equal( "1", ts.GetHists[0].as( "TH1F" ).GetTitle )
    assert_equal( 2, ts.GetHists.length )
  end
  # Test for regression in AddEntry ( present in r21520 )
  def test_legend_addentry
    # This works, despite After being used first, returning a TObject, and
    # taking one.
    list = TList.new
    p1 = TH1F.new( rand.to_s, "1", 5, 0, 5 )
    p2 = TH1F.new( rand.to_s, "2", 5, 0, 5 )
    p3 = TH1F.new( rand.to_s, "3", 5, 0, 5 )
    list.Add( p1 )
    list.Add( p2 )
    list.Add( p3 )
    assert_equal( "2", list.After( p1 ).as( "TH1F" ).GetTitle )

    # This does not work with this regression
    legend = TLegend.new
    p1 = TH1F.new( rand.to_s, rand.to_s, 5, 0, 5 ) 
    p2 = TH1F.new( rand.to_s, rand.to_s, 5, 0, 5 ) 
    legend.AddEntry( p1, 1.to_s )
    legend.AddEntry( p2, 2.to_s )
    # Apparently the first AddEntry does not work.
    assert_equal( "1", legend.GetListOfPrimitives[0].as( "TLegendEntry" ).GetLabel )
    assert_equal( 2, legend.GetListOfPrimitives.length )
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
  # Check that casting via "as" works
  def test_as_casting
    tlv = nil
    assert_nothing_raised do
      tlv = TLorentzVector.new.as( "TObject" )
    end
    assert_equal( TObject, tlv.class )
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
  # Test for regression in calling ( char* ) methods with fixnum arguments
#  def test_method_arg_error_regression
#    legend = TLegend.new
#    p1 = TH1F.new( rand.to_s, rand.to_s, 5, 0, 5 )
#    assert_raise( ArgumentError ) do
#      legend.AddEntry( p1, 1 )
#    end
#  end
  # Test for error in Accessing THStack#GetXaxis
  # fixed in r21541
  # Also tests for Draw without histograms regression (r28092) which caused a
  # segfault. See https://savannah.cern.ch/bugs/index.php?53803
  def test_thstack_getxaxis
    ts = THStack.new
    assert_nil( ts.GetXaxis )
    ts.Draw
    assert_nil( ts.GetXaxis )
    ts.Add( TH1F.new( rand.to_s, rand.to_s, 5, 0, 5 ) ) 
    ts.Draw
    assert( ts.GetXaxis, 'GetXaxis is nil' )
    assert_equal( TAxis, ts.GetXaxis.as( "TAxis" ).class )
  end
  # Problem introduced in r26567: method_missing of Object is changed, yielding
  # unexpected behaviour in external classes. I.e. failing private method calls
  # (handled via singleton method method_missing) are passed to Root, which
  # they should not. Possible workaround: Move this into DRRAbstractClass,
  # where it only affects ROOT Objects. Disadvantage: This still has the
  # problem of not allowing you to make methods private.
  class NonRootObject
    def NonRootObject.test
    end
    private_class_method :test
    private
    def test
    end
  end
  def test_non_root_objects_are_unaffected
    assert_raise( NoMethodError ) do
      nro = NonRootObject.new
      nro.test
    end
    assert_raise( NoMethodError ) do
      NonRootObject.test
    end
  end
  # Make sure that TMath's methods can now be called. Works since a singleton
  # method_missing is implemented.
  def test_singleton_method_missing
    assert_in_delta( TMath.E, Math::exp( 1 ), 1e-5 )
  end
end
