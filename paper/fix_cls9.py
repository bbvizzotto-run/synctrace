"""v9: comment the 3 broken hook lines, add xcolor after wrapfig require."""
s = open("sn-jnl.cls.bak").read()

old = "\\SetFootnoteHook{\\hspace*{-8pt}}%\n\\DeclareNewFootnote{A}[gobble]%\n\\setlength{\\skip\\footinsA}{0pt}}%\n"
new = "%\\SetFootnoteHook{\\hspace*{-8pt}}%\n%\\DeclareNewFootnote{A}[gobble]%\n%\\setlength{\\skip\\footinsA}{0pt}%\n}%\n"
assert old in s, "hook not found"
s = s.replace(old, new)

marker = "\\RequirePackage{wrapfig}%"
i = s.find(marker)
assert i >= 0, "wrapfig marker not found"
s = s[:i + len(marker)] + "\n\\RequirePackage{xcolor}" + s[i + len(marker):]

open("sn-jnl.cls", "w").write(s)
print("v9 applied")
