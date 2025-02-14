#!/bin/bash
# The Python tutorials appear in the Namespaces page. This script removes them from Namespaces.

# defime HTMLPATH
HTMLPATH=$DOXYGEN_OUTPUT_DIRECTORY/html
if [ ! -d "$HTMLPATH" ]; then
   echo "Error: DOXYGEN_OUTPUT_DIRECTORY is not exported."
   exit 1
fi

# change __ to _ in the input parameter
u=${1//__/_}

# clean namespaces in $HTMLPATH and in $HTMLPATH/search
echo "Clean namespace for" $u
s="namespace$1.html"
find "$HTMLPATH" -type f | xargs -P 12 -n 100 grep -s -l "$s" | while IFS= read -r file; do
  if test -e "$file"; then
	 echo "   Patching:" $file
	 # Remove the links to the namespace
	 sed -e "s/<a href=.$s.>$u<\/a>/$u/" "$file" > "$HTMLPATH/TMP_FILE"
	 mv "$HTMLPATH/TMP_FILE" "$file"
	 sed -e "s/<a c.*href=.$s.>$u<\/a>/$u/" "$file" > "$HTMLPATH/TMP_FILE"
	 mv "$HTMLPATH/TMP_FILE" "$file"
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

# remove the faulty namespace file
file=$HTMLPATH/namespace$1.html
if test -e "$file"; then
   rm $file
fi
