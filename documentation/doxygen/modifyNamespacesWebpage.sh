#!/bin/bash
# The Python tutorials appear in the Namespaces page. 
# This script removes the namespace given as input from all the files.

# defime HTMLPATH
HTMLPATH=$DOXYGEN_OUTPUT_DIRECTORY/html
if [ ! -d "$HTMLPATH" ]; then
   echo "Error: DOXYGEN_OUTPUT_DIRECTORY is not exported."
   exit 1
fi

u="${1//__/_}"
echo "Clean namespace for" $u
s="namespace$1.html"

# remove the faulty namespace file
file=$HTMLPATH/namespace$1.html
if test -e "$file"; then
   echo "   Remove:" $file
   rm $file
fi

# clean namespace in namespaces.html
file=$HTMLPATH/namespaces.html
ln=$(grep -n "$s" "$file" | cut -d: -f1)
if [ -n "$ln" ]; then
   echo "   Patching:" $file
   sed -e "/$s/d" "$file" > "$HTMLPATH/TMP_FILE"
   mv "$HTMLPATH/TMP_FILE" "$file"
fi

# clean namespace in doxygen_crawl.html
file=$HTMLPATH/doxygen_crawl.html
ln=$(grep -n "$s" "$file" | cut -d: -f1)
if [ -n "$ln" ]; then
   echo "   Patching:" $file
   sed -e "/$s/d" "$file" > "$HTMLPATH/TMP_FILE"
   mv "$HTMLPATH/TMP_FILE" "$file"
fi

# clean namespace in classes.html
file=$HTMLPATH/classes.html
ln=$(grep -n "$s" "$file" | cut -d: -f1)
if [ -n "$ln" ]; then
   echo "   Patching:" $file
   sed -e "/$s/d" "$file" > "$HTMLPATH/TMP_FILE"
   mv "$HTMLPATH/TMP_FILE" "$file"
fi

# clean namespace in annotated.html
file=$HTMLPATH/annotated.html
i=$(grep "$s" "$file" | sed -n 's/.*<tr[^>]*id="\([^"]*\)".*/\1/p')
if [ -n "$i" ]; then
   echo "   Patching:" $file
   sed "/id=\"$i/d" "$file" > "$HTMLPATH/TMP_FILE"
   mv "$HTMLPATH/TMP_FILE" "$file"
fi

# clean namespace in ROOT.tag
file=$HTMLPATH/ROOT.tag
sed -e '/<compound kind="namespace">/,/<\/compound>/ {
  /<compound kind="namespace">/ {
    :loop
    N
    /<\/compound>/! b loop
    /<name>'"$u"'<\/name>/ d
  }
}' "$file" > "$HTMLPATH/TMP_FILE"
mv "$HTMLPATH/TMP_FILE" "$file"
   	 
# clean namespace in $HTMLPATH/search
find "$HTMLPATH/search" -type f | xargs -P 12 -n 100 grep -s -l "$s" | while IFS= read -r file; do
  if test -e "$file"; then
	 echo "   Patching:" $file
	 # Remove the line containing the namespace
	 sed -e "/$s/d" "$file" > "$HTMLPATH/TMP_FILE"
	 mv "$HTMLPATH/TMP_FILE" "$file"
  fi
done

# remove references to namespace in $HTMLPATH 
find "$HTMLPATH" -type f | xargs -P 12 -n 100 grep -s -l "$s" | while IFS= read -r file; do
  if test -e "$file"; then
	 echo "   Patching:" $file
	 # Remove the links to the namespace
     sed -e "s/<a class.*href=.$s.*>\(.*\)<\/a>/\1/" "$file" > "$HTMLPATH/TMP_FILE"
	 mv "$HTMLPATH/TMP_FILE" "$file"
  fi
done

# remove references to namespace in $HTMLPATH/*.js 
find "$HTMLPATH" -type f -name "*.js" | xargs -P 12 -n 100 grep -s -l "$s" | while IFS= read -r file; do
  if test -e "$file"; then
	 echo "   Patching:" $file
     # Remove the line containing the namespace
	 sed -e "/$s/d" "$file" > "$HTMLPATH/TMP_FILE"
	 mv "$HTMLPATH/TMP_FILE" "$file"
  fi
done

# clean memberdecls in the python file
file="$HTMLPATH/$1_8py.html"
if test -e "$file"; then
   sed -e "/memberdecls/,+5d" "$file" > "$HTMLPATH/TMP_FILE"
   mv "$HTMLPATH/TMP_FILE" "$file"
fi
