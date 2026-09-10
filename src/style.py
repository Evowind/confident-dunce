"""Style partage des figures : chaque serie est distinguable par trois canaux redondants,
couleur (palette Okabe-Ito, sure pour les daltonismes), marqueur et style de trait, donc
lisible en noir et blanc. Les legendes ont des traits longs pour que le style soit visible.

    from .style import style_for, legend, hatch_for
    ax.plot(x, y, label=name, **style_for(name))
    legend(ax)
"""
import matplotlib.pyplot as plt

# Okabe & Ito (2008)
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE, GREY = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#8c8c8c")

# (mot-cle dans le nom -> couleur, marqueur, trait). Ordre = priorite : le premier qui matche gagne.
_RULES = [
    ("tckd",        dict(color=PURPLE,     marker="x", linestyle="-.")),
    ("nckd",        dict(color=ORANGE,     marker="v", linestyle="-")),
    ("kd_T1",       dict(color=SKY,        marker=">", linestyle=":")),
    ("kd_T2",       dict(color=VERMILLION, marker="<", linestyle="-.")),
    ("kd_T8",       dict(color=VERMILLION, marker="*", linestyle="--")),
    ("kd_ce05",     dict(color=PURPLE,     marker="p", linestyle="-")),
    ("dkd",         dict(color=BLUE,       marker="D", linestyle="-")),
    ("feat",        dict(color=GREEN,      marker="P", linestyle="-")),
    ("kd",          dict(color=VERMILLION, marker="o", linestyle="-")),
    ("scratch",     dict(color=BLACK,      marker="s", linestyle="--")),
    ("teacher",     dict(color=GREY,       marker="^", linestyle=":")),
]
# variantes "robustes" (teacher AugMix / fort, images augmentees) : memes reperes, marqueur creux
_HOLLOW_KEYS = ("augmix", "strong", "_aug")


def style_for(name: str, hollow=None) -> dict:
    """Style complet pour une serie nommee (nom de run ou de config)."""
    n = name.lower()
    st = dict(color=GREY, marker="o", linestyle="-")
    for key, s in _RULES:
        if key.lower() in n:
            st = dict(s)
            break
    if hollow is None:
        hollow = any(k in n for k in _HOLLOW_KEYS)
    st.update(markersize=7, linewidth=2, markeredgewidth=1.6)
    if hollow:   # variante robuste : marqueur creux ET trait tirete (deux reperes independants de la couleur)
        st.update(markerfacecolor="white", markeredgecolor=st["color"])
        if st["linestyle"] == "-":
            st["linestyle"] = "--"
    return st


def hatch_for(name: str) -> str:
    """Hachure pour les barres : redondante avec la couleur."""
    n = name.lower()
    for key, h in [("tckd", "xx"), ("nckd", ""), ("dkd", ".."), ("feat", "++"), ("kd", "//"), ("scratch", ""), ("teacher", "\\\\")]:
        if key in n:
            return h
    return ""


def color_for(name: str) -> str:
    return style_for(name)["color"]


def legend(ax, **kw):
    """Legende avec traits longs (le style de trait devient lisible) et marqueur visible."""
    opts = dict(handlelength=4.0, numpoints=1, markerscale=1.1, fontsize=8, frameon=True, framealpha=0.9)
    opts.update(kw)
    return ax.legend(**opts)


def apply_rc():
    plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=[BLUE, VERMILLION, ORANGE, GREEN, PURPLE, SKY, BLACK, GREY]),
                         "lines.linewidth": 2, "legend.handlelength": 4.0})
