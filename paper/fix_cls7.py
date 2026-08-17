"""Clean cls fix v7: comment hook lines, add xcolor after ACTIVE geometry."""
s = open("sn-jnl.cls.bak").read()

# Neutralize the 3 hook lines.
old = "\\SetFootnoteHook{\\hspace*{-8pt}}%\n\\DeclareNewFootnote{A}[gobble]%\n\\setlength{\\skip\\footinsA}{0pt}}%\n"
new = "%\\SetFootnoteHook{\\hspace*{-8pt}}%\n%\\DeclareNewFootnote{A}[gobble]%\n%\\setlength{\\skip\\footinsA}{0pt}}%\n"
assert old in s, "hook lines not found"
s = s.replace(old, new)

# Insert xcolor after the ACTIVE geometry require (skip the commented one).
active = "\\RequirePackage{geometry}"
idxs = [i for i in range(len(s)) if s.startswith(active, i)]
assert len(idxs) >= 2, f"expected >=2 geometry occurrences, got {len(idxs)}"
i = idxs[-1]  # active occurrence is later than the commented one
assert s[max(0, i - 2):i] != "%%", f"unexpected commented at {i}"
s = s[:i + len(active)] + "\n\\RequirePackage{xcolor}" + s[i + len(active):]

open("sn-jnl.cls", "w").write(s)
print("v7 applied, xcolor after active geometry at", i)
