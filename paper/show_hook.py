s = open("manuscript.log").read()
i = s.find("\\SetFootnoteHook")
seg = s[i:i+300]
print(repr(seg))
