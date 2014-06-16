# Make the ouput directory for the doxygen output
outdir=`grep "^OUTPUT_DIRECTORY" Doxyfile | sed -e "s/^.*= //"`

if [ ! -d "$outdir" ]
then
   mkdir -p $outdir
fi

echo $outdir
