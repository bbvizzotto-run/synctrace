import re

p = "sn-jnl.cls"
s = open("sn-jnl.cls.bak").read()

# 1) Add xcolor requirement near the other Requires (footmisc not needed if we
#    provide the macros ourselves before use).
assert "\\RequirePackage{geometry}" in s
s = s.replace("\\RequirePackage{geometry}",
              "\\RequirePackage{xcolor}%\n\\RequirePackage{geometry}")

# 2) In the AtBeginDocument footnote hook, provide DeclareNewFootnote first,
#    and guard the skip assignment.
old_hook = ("SetFootnoteHook{\\hspace*{-8pt}}%\n"
            "\\DeclareNewFootnote{A}[gobble]%\n"
            "\\setlength{\\skip\\footinsA}{0pt}}%")
assert old_hook in s
new_hook = ("SetFootnoteHook{\\hspace*{-8pt}}%\n"
            "\\providecommand{\\DeclareNewFootnote}[2][]{}\n"
            "\\DeclareNewFootnote{A}[gobble]%\n"
            "\\ifcsname footinsA\\endcsname\\setlength{\\skip\\footinsA}{0pt}\\fi}%")
s = s.replace(old_hook, new_hook)

open(p, "w").write(s)
print("cls fixed (xcolor + conditional footnote defs)")
