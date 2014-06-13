# Make the ouput directory for the macros
outdir=`grep "^EXAMPLE_PATH" Doxyfile | sed -e "s/^.*= //"`

if [ ! -d "$outdir" ]
then
   mkdir -p $outdir
fi

echo $outdir
