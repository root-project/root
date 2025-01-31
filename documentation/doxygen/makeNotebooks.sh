#!/bin/bash
# Run commands to generate notebooks, possibly in parallel.
# List dependencies below as <tutorial> <tutorial it depends on>


while [ $# -gt 0 ]; do
  case $1 in
    -j*)
      nJobs=${1#-j}
      shift
      ;;
    *)
      inputFile=$1
      shift
      ;;
  esac
done

if [ ! -f "$inputFile" ]; then
  echo "Usage: makeNotebooks <fileWithNotebooks to generate> [-j jobs]"
  exit 1
fi

# Prepare jupyter runs
mkdir -p ~/.jupyter

# If notebooks depend on the output of others, run those first.
# List dependencies below in the form
# <Notebook with dependencies> <dependency> [<dependency> ...]
while read notebook dependencies; do
	if grep -q ${notebook} $inputFile; then
    for dependency in $dependencies; do
      cmd=$(grep $dependency $inputFile)
      if [ -n "$cmd" ]; then
        echo "Running $cmd as depedency of $notebook"
        ${Python3_EXECUTABLE:-python3} $cmd  && sed -i'.back' "\#${dependency}#d" $inputFile
      fi
    done
  fi
done <<EOF
analysis/unfold/testUnfold5d.C analysis/unfold/testUnfold5a.C
analysis/unfold/testUnfold5b.C analysis/unfold/testUnfold5c.C
io/xml/xmlreadfile.C io/xml/xmlnewfile.C
roofit/roofit/rf503_wspaceread.C roofit/roofit/rf502_wspacewrite.C roofit/roofit/rf502_wspacewrite.py
io/readCode.C io/importCode.C
math/fit/fit1.C hist/hist001_TH1_fillrandom.C
math/fit/myfit.C math/fit/fitslicesy.C
math/foam/foam_demopers.C math/foam/foam_demo.C
io/tree/tree500_cernbuild.C io/tree/tree501_cernstaff.C io/tree/tree502_staff.C
hist/hist006_TH1_bar_charts.C io/tree/ntuple1.py hsimple.py
math/fit/fit1.py hist/hist001_TH1_fillrandom.py
machine_learning/TMVAClassificationApplication.C machine_learning/TMVAClassification.C
machine_learning/TMVAClassificationCategory.C machine_learning/TMVAClassification.C
machine_learning/TMVAClassificationCategoryApplication.C machine_learning/TMVAClassificationCategory.C
machine_learning/TMVAMulticlass.C machine_learning/TMVAMultipleBackgroundExample.C
machine_learning/TMVAMulticlassApplication.C machine_learning/TMVAMulticlass.C
machine_learning/TMVARegressionApplication.C machine_learning/TMVARegression.C
EOF

# Run rest in parallel
xargs -L 1 -P ${nJobs:-1} ${Python3_EXECUTABLE:-python3} < $inputFile

rm ${inputFile}.back

