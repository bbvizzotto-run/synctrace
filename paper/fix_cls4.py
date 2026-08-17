"""Final cls fix: disable the broken footnote hook lines and add xcolor."""
s = open("sn-jnl.cls.bak").read()

# 1) Neutralize the three broken hook lines inside \AtBeginDocument by
#    commenting them out with % (they come from footmisc which the class
#    forgets to require).
old_hook = ("SetFootnoteHook{\\hspace*{-8pt}}%\n"
            "\\DeclareNewFootnote{A}[gobble]%\n"
            "\\setlength{\\skip\\footinsA}{0pt}}%")
assert old_hook in s
new_hook = ("%%SetFootnoteHook{\\hspace*{-8pt}}% (disabled: footmisc unavailable)\n"
            "%%\\DeclareNewFootnote{A}[gobble]%\n"
            "%%\\setlength{\\skip\\footinsA}{0pt}}%")
s = s.replace(old_hook, new_hook)

# 2) Add xcolor: the class uses \definecolor without requiring xcolor.
#    Find an uncommented RequirePackage and insert xcolor right after it.
if "\\RequirePackage{xcolor}" not in s:
    s = s.replace("\\RequirePackage{geometry}",
                  "\\RequirePackage{xcolor}%\n\\RequirePackage{geometry}")

open("sn-jnl.cls", "w").write(s)
print("final patch applied")
