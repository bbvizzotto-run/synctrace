"""Clean cls fix v5, starting from the original backup."""
s = open("sn-jnl.cls.bak").read()

# Neutralize the 3 hook lines (full-line replacement).
old = "\\SetFootnoteHook{\\hspace*{-8pt}}%\n\\DeclareNewFootnote{A}[gobble]%\n\\setlength{\\skip\\footinsA}{0pt}}%\n"
new = "%\\SetFootnoteHook{\\hspace*{-8pt}}%\n%\\DeclareNewFootnote{A}[gobble]%\n%\\setlength{\\skip\\footinsA}{0pt}}%\n"
assert old in s, "hook lines not found as expected"
s = s.replace(old, new)

# Require xcolor (used by \definecolor in the class).
assert "\\RequirePackage{geometry}" in s and "\\RequirePackage{xcolor}" not in s
s = s.replace("\\RequirePackage{geometry}",
              "\\RequirePackage{xcolor}%\n\\RequirePackage{geometry}")

open("sn-jnl.cls", "w").write(s)
print("v5 applied")
