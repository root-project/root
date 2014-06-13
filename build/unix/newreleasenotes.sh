#! /bin/sh

PACKAGES="core io net sql tree proof hist interpreter bindings math roofit \
          tmva geom montecarlo gui graf2d graf3d html misc tutorials"

VERS="602"

OUTDIR="README/ReleaseNotes"

echo ""
echo "Creating new package docs..."


# write package notes
for i in $PACKAGES; do
   for v in $VERS; do
      if [ ! -r ${i}/doc/v${v}/index.md ]; then
         mkdir ${i}/doc/v${v}
         touch ${i}/doc/v${v}/index.md
         echo "created ${i}/doc/v${v}/index.md"
      fi
   done
done

echo ""
echo "Generated v${v}/index.md (s)"
echo ""

exit 0
