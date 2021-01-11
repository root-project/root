/// \file
/// \ingroup tutorial_graphs
/// \notebook -js
/// Plot a graph to visualize Time Dilation.
///
/// Time Dilation is a phenomena that was predicted by the Lorentz factor.
/// Albert Einstein later confirmed that this effect concerns the nature of
/// time itself in his Special Theory of Relativity. 
///
/// The phenomena states, that time is not absolute and that the time percieved 
/// by an observer is dependant on their relative velocity. So two observers can
/// observe the same event differently due to the difference in their velocities. 

/// In this program, you will see that the plotted graph shows how long a second 
/// is measured by a stationary observer (or an inertial observer) relative to a
/// second observer travelling at a certain percentage of the speed of light.
///
/// As you can see, the graph is undefined at 100% the speed of light, as one must
/// divide by zero. The laws of Physics break down here.
///
/// \macro_image
/// \macro_code
///
/// \author Advait Dhingra



double calculate(double v); // unit is m/s

void timeDilation()
{
	TGraph *gr = new TGraph();
	
	
	for (int v = 0; v < 100; v++) // velocity increases by one every iteration
	{
		double td = calculate(v);
		gr->SetPoint(gr->GetN(), v, td);
	}
	
	TCanvas *c1 = new TCanvas();
	
	gr->GetXaxis()->SetTitle("Percent of the speed of light");
	gr->GetYaxis()->SetTitle("1 second as measured by stationary observer");
	
	gr->Draw("AL");
	
}
double calculate(double v)
{
	double td = 1 / sqrt(1 - (v / 100)); // the time dilation formula
	
	return td;
}
