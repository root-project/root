#include "TChain.h"

int execLoadDim() {
   TChain c("Nominal/llllTree");
   c.Add("loadDim.root");
   c.SetScanField(0); // Show all rows without asking for CR or quit interactively
   auto entries = c.Scan("((llll_type&&(llll_sumz_rank)==MinIf$(llll_sumz_rank,(((((((((runNumber>=297730)&&(passCleaning&&passNPV))&&(passTriggers&((runNumber<=284484)*435 + (runNumber<=302087)*76 + (runNumber>=296639)*44032 + (runNumber>=266639&&runNumber<=302393)*65536 + 20992)))&&(llll_charge==0&&llll_dCharge==0))&&((l_quality[llll_l1]*(abs(l_pdgId[llll_l1])==11)!=2&&l_quality[llll_l2]*(abs(l_pdgId[llll_l2])==11)!=2&&l_quality[llll_l3]*(abs(l_pdgId[llll_l3])==11)!=2)&&(l_quality[llll_l4]*(abs(l_pdgId[llll_l4])==11)!=2)))&&(!(llll_overlaps&54)))&&(l_tlv_pt[llll_l1]>20000&&l_tlv_pt[llll_l2]>15000&&l_tlv_pt[llll_l3]>10000))&&(llll_triggerMatched))&&(llll_nCTorSA<2))&&(llll_type)))&&(((((llll_min_sf_dR>0.1&&llll_min_of_dR>0.2)&&(llll_max_el_d0Sig<5&&llll_max_mu_d0Sig<3))&&(llll_l_isIsolFixedCutLoose==15))&&(!(fabs(ll_tlv_m[llll_ll1])<5000.||fabs(ll_tlv_m[llll_ll2])<5000.||fabs(ll_tlv_m[llll_alt_ll1])<5000.||fabs(ll_tlv_m[llll_alt_ll2])<5000.)))&&(ll_tlv_m[llll_ll1]>66000&&ll_tlv_m[llll_ll1]<116000&&ll_tlv_m[llll_ll2]>66000&&ll_tlv_m[llll_ll2]<116000)))&&((runNumber>=297730)&&(Sum$(((((truth_llll_truePair)&&(truth_llll_min_el_pt>7000.&&truth_llll_min_mu_pt>7000.))&&(truth_llll_max_el_eta<2.5&&truth_llll_max_mu_eta<2.7)))*truth_llll_pdgIdSum)==44))");
   assert (entries == 10);
   return 0;
}
