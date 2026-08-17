"""Clean cls fix v6: insert xcolor after the ACTIVE geometry require."""
s = open("sn-jnl.cls.bak").read()

# Neutralize the 3 hook lines.
old = "\\SetFootnoteHook{\\hspace*{-8pt}}%\n\\DeclareNewFootnote{A}[gobble]%\n\\setlength{\\skip\\footinsA}{0pt}}%\n"
new = "%\\SetFootnoteHook{\\hspace*{-8pt}}%\n%\\DeclareNewFootnote{A}[gobble]%\n%\\setlength{\\skip\\footinsA}{0pt}}%\n"
assert old in s, "hook lines not found"
s = s.replace(old, new)

# Insert xcolor AFTER the active (uncommented) \RequirePackage{geometry}.
active = "\\RequirePackage{geometry}"
commented = "%%\\RequirePackage{geometry}"
assert s.count(active) >= 1
i = s.find(active)
# ensure we are not inside a commented line
assert s[max(0, i - 2):i] != "%%", f"matched commented occurrence at {i}"
s = s[:i + len(active)] + "\n\\RequirePackage{xcolor}" + s[i + len(active):]

open("sn-jnl.cls", "w").write(s)
print("v6 applied")
