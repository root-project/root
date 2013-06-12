#! /bin/sh

PACKAGES="core io net sql tree proof hist cint bindings math roofit \
          tmva geom montecarlo gui graf2d graf3d html misc tutorials"

VERS="534"

OUTDIR="README/ReleaseNotes"

echo ""
echo "Generating $OUTDIR from package docs..."
echo ""

# make output dir
if [ ! -d $OUTDIR ]; then
   mkdir $OUTDIR
fi

# make clean version directories
for v in $VERS; do
   if [ -d $OUTDIR/v${v} ]; then
      rm -rf $OUTDIR/v${v}
   fi
   mkdir $OUTDIR/v${v}
done

# write header note
for v in $VERS; do
   if [ -r doc/v${v}/index.html ]; then
      cat doc/v${v}/index.html > $OUTDIR/v${v}/index.html
   fi
done

# write package notes
for i in $PACKAGES; do
   for v in $VERS; do
      if [ -r ${i}/doc/v${v}/index.html ]; then
         echo "<!-- ${i}/doc/v${v}/index.html -->" >> $OUTDIR/v${v}/index.html
         cat ${i}/doc/v${v}/index.html >> $OUTDIR/v${v}/index.html
         for img in `ls ${i}/doc/v${v}/*`; do 
            if [ $img != ${i}/doc/v${v}/index.html ]; then
               cp ${img} $OUTDIR/v${v}
            fi
         done
      fi
   done
done

# write trailer note
echo ""
for v in $VERS; do
   if [ -r doc/v${v}/Trailer.html ]; then
      echo "<!-- doc/v${v}/Trailer.html -->" >> $OUTDIR/v${v}/index.html
      cat doc/v${v}/Trailer.html >> $OUTDIR/v${v}/index.html
      echo "Generated $OUTDIR/v${v}/index.html"
      echo " "
      echo "Check the generated file with:"
      echo "   http://validator.w3.org/#validate_by_upload+with_options"
   fi
done
echo ""

exit 0
