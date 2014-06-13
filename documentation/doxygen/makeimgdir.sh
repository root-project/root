# Make the ouput directory for the images
imgdir=`grep "^IMAGE_PATH" Doxyfile | sed -e "s/^.*= //"`

if [ ! -d "$imgdir" ]
then
   mkdir -p $imgdir
fi

echo $imgdir
