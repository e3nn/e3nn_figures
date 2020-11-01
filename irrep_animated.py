# pylint: disable=not-callable, no-member, invalid-name, missing-docstring, line-too-long
import argparse
import os
import shutil
import subprocess

import plotly.graph_objs as go
import torch
from tqdm.auto import tqdm
from e3nn.tensor import IrrepTensor, SphericalTensor


def get_cmap(x):
    if x == 'bwr':
        return [[0, 'rgb(0,50,255)'], [0.5, 'rgb(200,200,200)'], [1, 'rgb(255,50,0)']]
    if x == 'plasma':
        return [[0, '#9F1A9B'], [0.25, '#0D1286'], [0.5, '#000000'], [0.75, '#F58C45'], [1, '#F0F524']]


def surf(args, x, center):
    x = IrrepTensor(x, len(x) // 2)
    x = SphericalTensor.from_irrep_tensor(x)

    return go.Surface(
        **x.plotly_surface(args.res, center=center, normalization='component'),
        showscale=False,
        cmin=-0.33,
        cmax=0.33,
        colorscale=get_cmap(args.cmap),
    )


def main(args):
    if os.path.exists('gif'):
        shutil.rmtree('gif')
    os.makedirs('gif')

    xs = torch.randn(args.pitchs, 2 * args.L + 1)

    for i, t in enumerate(tqdm(torch.linspace(0, 1, args.steps + 1)[:-1])):
        t = t.item()
        j = round(t // (1 / args.pitchs))
        t = (t * args.pitchs) % 1
        x0 = xs[j]
        x1 = xs[j + 1] if j + 1 < args.pitchs else xs[0]
        x = x0 * (1 - t) + x1 * t

        data = [
            surf(args, x, torch.tensor([0, 0, 0.0])),
        ]

        axis = dict(
            showbackground=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title='',
            nticks=3,
            range=[-3, 3]
        )

        layout = dict(
            width=args.height,
            height=args.height,
            scene=dict(
                xaxis=dict(
                    **axis,
                ),
                yaxis=dict(
                    **axis,
                ),
                zaxis=dict(
                    **axis,
                ),
                aspectmode='manual',
                aspectratio=dict(x=1.8, y=1.8, z=1.8),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=-5, z=0),
                    projection=dict(type='orthographic'),
                ),
            ),
            paper_bgcolor=args.color_bg,
            plot_bgcolor=args.color_bg,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        fig = go.Figure(data=data, layout=layout)
        fig.write_image('gif/{:03d}.png'.format(i))

    subprocess.check_output(["convert", "-delay", "3", "-loop", "0", "-dispose", "2", "gif/*.png", "output.gif"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--res", type=int, default=100)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--pitchs", type=int, default=4)
    parser.add_argument("--color_bg", type=str, default="rgba(0,0,0,0)")
    parser.add_argument("--color_text", type=str, default="rgb(255,255,255)")
    parser.add_argument("--cmap", type=str, default="plasma")
    parser.add_argument("--L", type=int, default=2)

    main(parser.parse_args())
