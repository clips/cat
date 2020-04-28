"""Plotting of attention distributions."""
from matplotlib import pyplot as plt


def plot_attention(attentions, texts):
    assert len(attentions) == len(texts)
    fig, axes = plt.subplots(len(attentions), 1, figsize=(5, 3))

    if len(attentions) == 1:
        axes = [axes]

    for idx, (att, txt) in enumerate(zip(attentions, texts)):
        ax = axes[idx]
        ax.imshow(att[None, :],
                  vmin=.0,
                  vmax=1.0,
                  cmap="Reds",
                  aspect="auto")
        ax.set_xticks(range(att.shape[0]))
        ax.set_xticklabels(txt, rotation=45)
        ax.set_yticks([])

        for idx, x in enumerate(att):
            ax.text(idx-.2, 0, str(x.round(2))[1:])

    fig.tight_layout()
    return fig
