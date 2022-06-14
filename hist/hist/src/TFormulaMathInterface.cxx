#include "v5/TFormulaPrimitive.h"
#include "TMath.h"

namespace ROOT {

   namespace v5 {
      
void TMath_GenerInterface(){
//
// Automatically generated code  - don't modify it
//
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Pi","TMath::Pi",(TFormulaPrimitive::GenFunc0)TMath::Pi));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::TwoPi","TMath::TwoPi",(TFormulaPrimitive::GenFunc0)TMath::TwoPi));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::PiOver2","TMath::PiOver2",(TFormulaPrimitive::GenFunc0)TMath::PiOver2));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::PiOver4","TMath::PiOver4",(TFormulaPrimitive::GenFunc0)TMath::PiOver4));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::InvPi","TMath::InvPi",(TFormulaPrimitive::GenFunc0)TMath::InvPi));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::RadToDeg","TMath::RadToDeg",(TFormulaPrimitive::GenFunc0)TMath::RadToDeg));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::DegToRad","TMath::DegToRad",(TFormulaPrimitive::GenFunc0)TMath::DegToRad));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::E","TMath::E",(TFormulaPrimitive::GenFunc0)TMath::E));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Ln10","TMath::Ln10",(TFormulaPrimitive::GenFunc0)TMath::Ln10));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::LogE","TMath::LogE",(TFormulaPrimitive::GenFunc0)TMath::LogE));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::C","TMath::C",(TFormulaPrimitive::GenFunc0)TMath::C));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Ccgs","TMath::Ccgs",(TFormulaPrimitive::GenFunc0)TMath::Ccgs));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::CUncertainty","TMath::CUncertainty",(TFormulaPrimitive::GenFunc0)TMath::CUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::G","TMath::G",(TFormulaPrimitive::GenFunc0)TMath::G));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Gcgs","TMath::Gcgs",(TFormulaPrimitive::GenFunc0)TMath::Gcgs));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::GUncertainty","TMath::GUncertainty",(TFormulaPrimitive::GenFunc0)TMath::GUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::GhbarC","TMath::GhbarC",(TFormulaPrimitive::GenFunc0)TMath::GhbarC));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::GhbarCUncertainty","TMath::GhbarCUncertainty",(TFormulaPrimitive::GenFunc0)TMath::GhbarCUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Gn","TMath::Gn",(TFormulaPrimitive::GenFunc0)TMath::Gn));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::GnUncertainty","TMath::GnUncertainty",(TFormulaPrimitive::GenFunc0)TMath::GnUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::H","TMath::H",(TFormulaPrimitive::GenFunc0)TMath::H));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Hcgs","TMath::Hcgs",(TFormulaPrimitive::GenFunc0)TMath::Hcgs));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::HUncertainty","TMath::HUncertainty",(TFormulaPrimitive::GenFunc0)TMath::HUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Hbar","TMath::Hbar",(TFormulaPrimitive::GenFunc0)TMath::Hbar));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Hbarcgs","TMath::Hbarcgs",(TFormulaPrimitive::GenFunc0)TMath::Hbarcgs));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::HbarUncertainty","TMath::HbarUncertainty",(TFormulaPrimitive::GenFunc0)TMath::HbarUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::HC","TMath::HC",(TFormulaPrimitive::GenFunc0)TMath::HC));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::HCcgs","TMath::HCcgs",(TFormulaPrimitive::GenFunc0)TMath::HCcgs));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::K","TMath::K",(TFormulaPrimitive::GenFunc0)TMath::K));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Kcgs","TMath::Kcgs",(TFormulaPrimitive::GenFunc0)TMath::Kcgs));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::KUncertainty","TMath::KUncertainty",(TFormulaPrimitive::GenFunc0)TMath::KUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Sigma","TMath::Sigma",(TFormulaPrimitive::GenFunc0)TMath::Sigma));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::SigmaUncertainty","TMath::SigmaUncertainty",(TFormulaPrimitive::GenFunc0)TMath::SigmaUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Na","TMath::Na",(TFormulaPrimitive::GenFunc0)TMath::Na));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::NaUncertainty","TMath::NaUncertainty",(TFormulaPrimitive::GenFunc0)TMath::NaUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::R","TMath::R",(TFormulaPrimitive::GenFunc0)TMath::R));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::RUncertainty","TMath::RUncertainty",(TFormulaPrimitive::GenFunc0)TMath::RUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::MWair","TMath::MWair",(TFormulaPrimitive::GenFunc0)TMath::MWair));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Rgair","TMath::Rgair",(TFormulaPrimitive::GenFunc0)TMath::Rgair));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Qe","TMath::Qe",(TFormulaPrimitive::GenFunc0)TMath::Qe));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::QeUncertainty","TMath::QeUncertainty",(TFormulaPrimitive::GenFunc0)TMath::QeUncertainty));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Sin","TMath::Sin",(TFormulaPrimitive::GenFunc10)TMath::Sin));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Cos","TMath::Cos",(TFormulaPrimitive::GenFunc10)TMath::Cos));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Tan","TMath::Tan",(TFormulaPrimitive::GenFunc10)TMath::Tan));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::SinH","TMath::SinH",(TFormulaPrimitive::GenFunc10)TMath::SinH));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::CosH","TMath::CosH",(TFormulaPrimitive::GenFunc10)TMath::CosH));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::TanH","TMath::TanH",(TFormulaPrimitive::GenFunc10)TMath::TanH));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ASin","TMath::ASin",(TFormulaPrimitive::GenFunc10)TMath::ASin));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ACos","TMath::ACos",(TFormulaPrimitive::GenFunc10)TMath::ACos));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ATan","TMath::ATan",(TFormulaPrimitive::GenFunc10)TMath::ATan));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ATan2","TMath::ATan2",(TFormulaPrimitive::GenFunc110)TMath::ATan2));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ASinH","TMath::ASinH",(TFormulaPrimitive::GenFunc10)TMath::ASinH));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ACosH","TMath::ACosH",(TFormulaPrimitive::GenFunc10)TMath::ACosH));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ATanH","TMath::ATanH",(TFormulaPrimitive::GenFunc10)TMath::ATanH));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Hypot","TMath::Hypot",(TFormulaPrimitive::GenFunc110)TMath::Hypot));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Sqrt","TMath::Sqrt",(TFormulaPrimitive::GenFunc10)TMath::Sqrt));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Ceil","TMath::Ceil",(TFormulaPrimitive::GenFunc10)TMath::Ceil));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Floor","TMath::Floor",(TFormulaPrimitive::GenFunc10)TMath::Floor));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Exp","TMath::Exp",(TFormulaPrimitive::GenFunc10)TMath::Exp));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Power","TMath::Power",(TFormulaPrimitive::GenFunc110)TMath::Power));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Log","TMath::Log",(TFormulaPrimitive::GenFunc10)TMath::Log));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Log2","TMath::Log2",(TFormulaPrimitive::GenFunc10)TMath::Log2));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Log10","TMath::Log10",(TFormulaPrimitive::GenFunc10)TMath::Log10));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Abs","TMath::Abs",(TFormulaPrimitive::GenFunc10)TMath::Abs));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Sign","TMath::Sign",(TFormulaPrimitive::GenFunc110)TMath::Sign));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Min","TMath::Min",(TFormulaPrimitive::GenFunc110)TMath::Min));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Max","TMath::Max",(TFormulaPrimitive::GenFunc110)TMath::Max));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Range","TMath::Range",(TFormulaPrimitive::GenFunc1110)TMath::Range));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BreitWigner","TMath::BreitWigner",(TFormulaPrimitive::GenFunc1110)TMath::BreitWigner));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BesselI0","TMath::BesselI0",(TFormulaPrimitive::GenFunc10)TMath::BesselI0));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BesselK0","TMath::BesselK0",(TFormulaPrimitive::GenFunc10)TMath::BesselK0));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BesselI1","TMath::BesselI1",(TFormulaPrimitive::GenFunc10)TMath::BesselI1));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BesselK1","TMath::BesselK1",(TFormulaPrimitive::GenFunc10)TMath::BesselK1));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BesselJ0","TMath::BesselJ0",(TFormulaPrimitive::GenFunc10)TMath::BesselJ0));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BesselJ1","TMath::BesselJ1",(TFormulaPrimitive::GenFunc10)TMath::BesselJ1));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BesselY0","TMath::BesselY0",(TFormulaPrimitive::GenFunc10)TMath::BesselY0));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BesselY1","TMath::BesselY1",(TFormulaPrimitive::GenFunc10)TMath::BesselY1));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::StruveH0","TMath::StruveH0",(TFormulaPrimitive::GenFunc10)TMath::StruveH0));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::StruveH1","TMath::StruveH1",(TFormulaPrimitive::GenFunc10)TMath::StruveH1));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::StruveL0","TMath::StruveL0",(TFormulaPrimitive::GenFunc10)TMath::StruveL0));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::StruveL1","TMath::StruveL1",(TFormulaPrimitive::GenFunc10)TMath::StruveL1));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Beta","TMath::Beta",(TFormulaPrimitive::GenFunc110)TMath::Beta));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BetaCf","TMath::BetaCf",(TFormulaPrimitive::GenFunc1110)TMath::BetaCf));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BetaDist","TMath::BetaDist",(TFormulaPrimitive::GenFunc1110)TMath::BetaDist));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BetaDistI","TMath::BetaDistI",(TFormulaPrimitive::GenFunc1110)TMath::BetaDistI));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::BetaIncomplete","TMath::BetaIncomplete",(TFormulaPrimitive::GenFunc1110)TMath::BetaIncomplete));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::CauchyDist","TMath::CauchyDist",(TFormulaPrimitive::GenFunc1110)TMath::CauchyDist));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::DiLog","TMath::DiLog",(TFormulaPrimitive::GenFunc10)TMath::DiLog));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Erf","TMath::Erf",(TFormulaPrimitive::GenFunc10)TMath::Erf));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ErfInverse","TMath::ErfInverse",(TFormulaPrimitive::GenFunc10)TMath::ErfInverse));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Erfc","TMath::Erfc",(TFormulaPrimitive::GenFunc10)TMath::Erfc));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::ErfcInverse","TMath::ErfcInverse",(TFormulaPrimitive::GenFunc10)TMath::ErfcInverse));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::FDist","TMath::FDist",(TFormulaPrimitive::GenFunc1110)TMath::FDist));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::FDistI","TMath::FDistI",(TFormulaPrimitive::GenFunc1110)TMath::FDistI));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Freq","TMath::Freq",(TFormulaPrimitive::GenFunc10)TMath::Freq));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Gamma","TMath::Gamma",(TFormulaPrimitive::GenFunc10)TMath::Gamma));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Gamma","TMath::Gamma",(TFormulaPrimitive::GenFunc110)TMath::Gamma));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::KolmogorovProb","TMath::KolmogorovProb",(TFormulaPrimitive::GenFunc10)TMath::KolmogorovProb));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::LandauI","TMath::LandauI",(TFormulaPrimitive::GenFunc10)TMath::LandauI));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::LaplaceDist","TMath::LaplaceDist",(TFormulaPrimitive::GenFunc1110)TMath::LaplaceDist));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::LaplaceDistI","TMath::LaplaceDistI",(TFormulaPrimitive::GenFunc1110)TMath::LaplaceDistI));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::LnGamma","TMath::LnGamma",(TFormulaPrimitive::GenFunc10)TMath::LnGamma));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::NormQuantile","TMath::NormQuantile",(TFormulaPrimitive::GenFunc10)TMath::NormQuantile));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Poisson","TMath::Poisson",(TFormulaPrimitive::GenFunc110)TMath::Poisson));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::PoissonI","TMath::PoissonI",(TFormulaPrimitive::GenFunc110)TMath::PoissonI));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Student","TMath::Student",(TFormulaPrimitive::GenFunc110)TMath::Student));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::StudentI","TMath::StudentI",(TFormulaPrimitive::GenFunc110)TMath::StudentI));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::Vavilov","TMath::Vavilov",(TFormulaPrimitive::GenFunc1110)TMath::Vavilov));
TFormulaPrimitive::AddFormula(new TFormulaPrimitive("TMath::VavilovI","TMath::VavilovI",(TFormulaPrimitive::GenFunc1110)TMath::VavilovI));
}

   } // end namespace v5

} // end namespace ROOT
      
