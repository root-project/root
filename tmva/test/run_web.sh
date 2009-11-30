echo "... Setup Root"
rm -f /afs/cern.ch/user/s/stelzer/public_html/tmva/?t`date +"%a"`.html  
rm -f ${HOME}/TMVA/www/?t*.html
cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.22.00/slc4_amd64_gcc34/root
. ./bin/thisroot.sh
echo "... Setup TMVA"
cd ${HOME}/TMVA
source ./setup.sh
echo "... Update from SVN"
#cd ./test;
#svn up 
#cd -
svn up ./test ./src/ ./execs/ ./macros/ 
cd ${HOME}/TMVA/src
echo "... Build TMVA library"
make | tee -a ../www/log_`date +"%a"`
cd ${HOME}/TMVA/execs
touch ${HOME}/TMVA/test/CompareHistsTrainAndApplied.C
echo "... Build TMVA executable and run the test" 
make web 2>&1 | tee -a ../www/log_`date +"%a"`
echo "... Copy local web space to http://cern.ch/stelzer/tmva" 
cp -v -r -u ${HOME}/TMVA/www/* /afs/cern.ch/user/s/stelzer/public_html/tmva  
