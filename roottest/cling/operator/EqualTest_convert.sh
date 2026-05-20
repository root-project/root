sed -e 's?FILE:.*[/\]?FILE:?' -e 's/.dll/.so/g' | sed -E "s,(: )-?[0-9]+$,\1N/A," \
| grep -v -e tagnum -e 'int c' -e '~privateOp2'
