#! /bin/sh

VERS="604"

DIR="README/ReleaseNotes"
pandoc -f markdown -t html -s -S -f markdown --toc -H documentation/users-guide/css/github.css --mathjax \
$DIR/v${v}/index.md -o $DIR/v${v}/index.html

echo "Generated $DIR/v${v}/index.html"
echo ""

exit 0
