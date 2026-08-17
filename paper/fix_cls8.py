"""v8: find how geometry is really loaded in the cls, then patch cleanly."""
s = open("sn-jnl.cls.bak").read()
import re
for m in re.finditer(r'.{60}geometry.{40}', s):
    print(repr(m.group()))
    print('---')
