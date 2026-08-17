import re
s = open("sn-jnl.cls").read()
for kw in ["definecolor", "RequirePackage", "usepackage", "LoadClass", "providecommand"]:
    idxs = [m.start() for m in __import__("re").finditer(re.escape(kw), s)]
    print(kw, len(idxs), "first:", idxs[:3])
