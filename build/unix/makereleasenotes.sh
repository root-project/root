#! /bin/sh

PACKAGES="core io net sql tree proof hist interpreter bindings math roofit \
          tmva geom montecarlo gui graf2d graf3d html misc tutorials"

VERS="602"

OUTDIR="README/ReleaseNotes"

echo ""
echo "Generating $OUTDIR from package docs..."

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
   if [ -r doc/v${v}/index.md ]; then
      cat doc/v${v}/index.md > $OUTDIR/v${v}/index.md
   fi
done

# write package notes
for i in $PACKAGES; do
   for v in $VERS; do
      if [ -r ${i}/doc/v${v}/index.md ]; then
         cat ${i}/doc/v${v}/index.md >> $OUTDIR/v${v}/index.md
         for img in `ls ${i}/doc/v${v}/*`; do 
            if [ $img != ${i}/doc/v${v}/index.md ]; then
               cp ${img} $OUTDIR/v${v}
            fi
         done
      fi
   done
done

# write trailer note
echo ""
for v in $VERS; do
   if [ -r doc/v${v}/Trailer.md ]; then
      cat doc/v${v}/Trailer.md >> $OUTDIR/v${v}/index.md
      echo "Generated $OUTDIR/v${v}/index.md"
   fi
done
echo ""

pandoc -f markdown -t html -s -S -f markdown --toc -H documentation/users-guide/css/github.css --mathjax \
$OUTDIR/v${v}/index.md -o $OUTDIR/v${v}/index.html

echo "Generated $OUTDIR/v${v}/index.html"
echo ""

exit 0
