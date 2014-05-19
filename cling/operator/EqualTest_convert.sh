sed -e 's?FILE:.*[/\]?FILE:?' -e 's/.dll/.so/g' \
| grep -v -e tagnum -e 'int c' -e '~privateOp2'
