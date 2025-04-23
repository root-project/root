{

const char * slash = "MC*.root";

TRegexp re(slash,kTRUE);
TString s = "MC_uds_reco-1.root";
s.Index(re);

}
