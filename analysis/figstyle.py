"""Text style shared by the generated figures of the paper: Times, as the text, at one size in print.

Each figure is drawn at the width at which the paper includes it (a fraction of TEXTWIDTH or COLUMNWIDTH), so the sizes
below are the sizes in print. FreeSerif is a TrueType Times (from URW Nimbus Roman, the Times of LaTeX's times package);
matplotlib embeds it cleanly, which it does not do for the OpenType (CFF) Nimbus Roman.
"""
import matplotlib
import matplotlib.pyplot as plt

FONT = ["Times New Roman", "Times", "FreeSerif", "Nimbus Roman", "DejaVu Serif"]
FS, FS_LETTER = 7, 8                       # pt in print: all text; bold panel letters
TEXTWIDTH, COLUMNWIDTH = 6.875, 3.25       # in, two-column layout of the paper


def use_fonts():
    plt.rcParams.update({"font.family": "serif", "font.serif": FONT, "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
                         "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "legend.title_fontsize": FS,
                         "figure.titlesize": FS})


def unify_text(fig, size=FS, keep=()):
    """Set every text of fig to `size`, except the texts in `keep` (e.g. the panel letters)."""
    for t in fig.findobj(matplotlib.text.Text):
        if t not in keep and t.get_text():
            t.set_fontsize(size)
