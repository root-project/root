{
TFile *f = TFile::Open("cond09_mc.000029.gen.COND._0002.pool.root");
f->MakeProject("libcond","*","RECREATE+nocompilation");
}
