import re

p = "sn-jnl.cls"
s = open(p).read()

# Revert our previous patch: restore the original three lines.
orig_hook = ("SetFootnoteHook{\\hspace*{-8pt}}%\n"
             "\\DeclareNewFootnote{A}[gobble]%\n"
             "\\setlength{\\skip\\footinsA}{0pt}}%")
patched = ("SetFootnoteHook{\\hspace*{-8pt}}%\n"
           "\\providecommand{\\DeclareNewFootnote}[2][]{}\n"
           "\\DeclareNewFootnote{A}[gobble]%\n"
           "\\ifcsname footinsA\\endcsname\\setlength{\\skip\\footinsA}{0pt}\\fi}%")
assert patched in s, "patched hook not found"
s = s.replace(patched, orig_hook)

# Add missing package requirements right after \LoadClass line.
i = s.find("\\LoadClass")
if "RequirePackage{xcolor}" not in s:
    # Insert RequirePackages right after the last RequirePackage block (geometry etc.)
    s = s.replace("\\RequirePackage{geometry}",
                  "\\RequirePackage{xcolor}%\n\\RequirePackage{footmisc}%\n\\RequirePackage{geometry}")

open(p, "w").write(s)
print("cls fixed with xcolor+footmisc requirements")
