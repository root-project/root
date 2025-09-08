/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
/// Example showing how progress bars work with Range operations in RDataFrame.
///
/// This tutorial demonstrates the correct way to use progress bars with Range()
/// operations using computationally intensive operations that show real progress.
/// The key point is that the progress bar must be attached AFTER the Range() 
/// to show the correct event count.
///
/// \macro_code
/// \macro_output

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TRandom.h>
#include <TMath.h>
#include <chrono>
#include <thread>
#include <cmath>

void df108_ProgressRange()
{
   std::cout << "=== RDataFrame Progress Bar with Range() - Complex Analysis ===" << std::endl;
   std::cout << "Learning how to correctly use progress bars with computationally intensive Range operations\n" << std::endl;
   
   // Example 1: Medium-scale physics simulation with complex calculations
   std::cout << "1. Physics Simulation: Range(0, 100000) from 400000 total events" << std::endl;
   {
      // Create a DataFrame with 400k events (reduced by 80%)
      ROOT::RDataFrame df(400000);
      
      // IMPORTANT: Apply Range FIRST, then attach progress bar
      // This ensures the progress bar shows 100000/100000, not 400000/400000
      auto ranged = df.Range(0, 100000);
      auto node = ROOT::RDF::AsRNode(ranged);
      ROOT::RDF::Experimental::AddProgressBar(node);
      
      // Complex physics simulation with multiple expensive operations
      auto physics = ranged.Define("particle_energy", []() {
                                // Simulate complex energy calculation
                                double energy = 0;
                                for(int i = 0; i < 400; ++i) {  // Increased for visible progress
                                   energy += TMath::Sqrt(TMath::Power(gRandom->Gaus(100, 20), 2) + 
                                                        TMath::Power(gRandom->Gaus(50, 10), 2));
                                }
                                return energy / 400.0;
                            })
                            .Define("momentum_x", [](double E) { 
                                // Complex momentum calculation
                                double px = 0;
                                for(int j = 0; j < 200; ++j) {  // Increased for visible progress
                                   px += TMath::Sin(E * j * 0.01) * TMath::Cos(j * 0.1);
                                }
                                return px / 200.0;
                            }, {"particle_energy"})
                            .Define("momentum_y", [](double E) { 
                                double py = 0;
                                for(int j = 0; j < 200; ++j) {  // Increased for visible progress
                                   py += TMath::Cos(E * j * 0.01) * TMath::Sin(j * 0.1);
                                }
                                return py / 200.0;
                            }, {"particle_energy"})
                            .Filter("particle_energy > 80.0")
                            .Define("invariant_mass", [](double E, double px, double py) {
                                // Calculate invariant mass with expensive operations
                                double mass_sq = E*E - px*px - py*py;
                                return TMath::Sqrt(TMath::Abs(mass_sq));
                            }, {"particle_energy", "momentum_x", "momentum_y"});
      
      auto count = physics.Count();
      auto energy_histo = physics.Histo1D("particle_energy");
      
      std::cout << "   → Events passing filter: " << *count << " / 100000 processed" << std::endl;
      std::cout << "   → Mean particle energy: " << energy_histo->GetMean() << " GeV" << std::endl;
   }
   
   // Example 2: Larger dataset - demonstrates real-time updates
   std::cout << "\n2. Larger example: Range(100000, 400000) from 1000000 events" << std::endl;
   {
      ROOT::RDataFrame df(1000000);
      
      // Process events 100000 to 400000 (300k events total)
      auto ranged = df.Range(100000, 400000);
      auto node = ROOT::RDF::AsRNode(ranged);
      ROOT::RDF::Experimental::AddProgressBar(node);
      
      // More complex analysis to show progress updates with heavy computation
      auto analysis = ranged.Define("detector_response", []() {
                                // Simulate complex detector response
                                double response = 0;
                                for(int layer = 0; layer < 30; ++layer) {  // Reduced complexity
                                   for(int channel = 0; channel < 15; ++channel) {  // Reduced complexity
                                      double signal = gRandom->Gaus(1.0, 0.1);
                                      response += TMath::Exp(-signal*signal) * 
                                                 TMath::Sin(layer * 0.1) * 
                                                 TMath::Cos(channel * 0.05);
                                   }
                                }
                                return response;
                            })
                            .Define("calibrated_energy", [](double response) {
                                // Complex calibration procedure
                                double calibrated = 0;
                                for(int iter = 0; iter < 60; ++iter) {  // Reduced complexity
                                   calibrated += response * TMath::Power(1.01, iter % 10) * 
                                                TMath::Log(1 + iter * 0.01);
                                }
                                return calibrated / 60.0;
                            }, {"detector_response"})
                            .Filter("calibrated_energy > 5.0")
                            .Define("analysis_weight", [](double energy) {
                                // Statistical weight calculation
                                double weight = 1.0;
                                for(int i = 0; i < 30; ++i) {  // Reduced complexity
                                   weight *= (1.0 + TMath::Sin(energy * i * 0.001));
                                }
                                return TMath::Log(weight);
                            }, {"calibrated_energy"});
      
      auto count = analysis.Count();
      auto energy_stats = analysis.Stats("calibrated_energy");
      
      std::cout << "   → Events after complex analysis: " << *count << " / 300000 processed" << std::endl;
      std::cout << "   → Mean calibrated energy: " << energy_stats->GetMean() << std::endl;
   }
   
   // Example 3: What happens if you attach progress bar BEFORE Range?
   std::cout << "\n3. Common mistake: Progress bar attached BEFORE Range" << std::endl;
   {
      ROOT::RDataFrame df(1000000);
      
      // WRONG WAY: Progress bar attached to original DataFrame
      auto node = ROOT::RDF::AsRNode(df);
      ROOT::RDF::Experimental::AddProgressBar(node);
      auto ranged = df.Range(0, 200000);  // This will show 1000000/1000000!
      
      // Even with complex operations, the counting will be wrong
      auto result = ranged.Define("complex_calc", []() { 
                               double result = 0;
                               for(int i = 0; i < 150; ++i) {  // Increased for visible progress
                                  result += TMath::Sin(i) * TMath::Cos(i * 0.1) * TMath::Log(1 + i);
                                  // Add some nested computation to slow it down
                                  for(int j = 0; j < 50; ++j) {
                                     result += TMath::Exp(-j * 0.01) * TMath::Tan(i * j * 0.001);
                                  }
                               }
                               return result;
                           })
                           .Count();
      
      std::cout << "   → Events processed: " << *result << std::endl;
      std::cout << "   → Problem: Progress bar showed 1000000/1000000 instead of 200000/200000!" << std::endl;
   }
   
}