"""Patch sn-jnl.cls locally to fix the broken footnote hook for pdflatex."""
p = "sn-jnl.cls"
s = open(p).read()
old = r"\AtBeginDocument{% \n%%\newcommand*\ExtraParaSkip{12pt}% \n\SetFootnoteHook{\hspace*{-8pt}}% \n\DeclareNewFootnote{A}[gobble]% \n\setlength{\skip\footinsA}{0pt}}"
# The actual text (line breaks may differ); replace a robust substring.
old = ("SetFootnoteHook{\\hspace*{-8pt}}%\n"
       "\\DeclareNewFootnote{A}[gobble]%\n"
       "\\setlength{\\skip\\footinsA}{0pt}}%")
if old not in s:
    # find approximate positions
    i = s.find("SetFootnoteHook")
    raise SystemExit(f"not found: {s[i-80:i+120]!r}")
new = (r"\ifdefined\SetFootnoteHook \SetFootnoteHook{\hspace*{-8pt}}\fi"
       "\n\\ifdefined\\DeclareNewFootnote"
       "\n  \\DeclareNewFootnote{A}[gobble]%"
       "\n  \\setlength{\\skip\\footinsA}{0pt}"
       "\n\\else%"
       "\n  \\newif\\if@footnoteAcompat%"
       "\n  \\@footnoteAcompatfalse%"
       "\n\\fi"
       "\n}")
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
