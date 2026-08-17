"""Fix sn-jnl.cls: unconditionally provide the missing footnote macros."""
p = "sn-jnl.cls"
s = open(p).read()

old = ("SetFootnoteHook{\\hspace*{-8pt}}%\n"
       "\\ifdefined\\DeclareNewFootnote\\DeclareNewFootnote{A}[gobble]%\n"
       "\\else\\providecommand{\\DeclareNewFootnote}[2][]{}\n"
       "\\DeclareNewFootnote{A}[gobble]\\fi%\n"
       "\\ifcsname footinsA\\endcsname\\setlength{\\skip\\footinsA}{0pt}\\fi}%")

# Provide the macros unconditionally before use.
new = ("SetFootnoteHook{\\hspace*{-8pt}}%\n"
       "\\providecommand{\\DeclareNewFootnote}[2][]{}\n"
       "\\DeclareNewFootnote{A}[gobble]%\n"
       "\\ifcsname footinsA\\endcsname\\setlength{\\skip\\footinsA}{0pt}\\fi}%")

assert old in s, "old not found"
s = s.replace(old, new)
open(p, "w").write(s)
print("patched unconditional")
