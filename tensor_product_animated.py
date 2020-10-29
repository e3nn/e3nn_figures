# pylint: disable=not-callable, no-member, invalid-name, missing-docstring, line-too-long
import argparse
import math
import os
import shutil
import subprocess

import plotly.graph_objs as go
import torch
from tqdm.auto import tqdm
from e3nn import o3, rs
from e3nn.tensor import IrrepTensor, SphericalTensor


def surf(x, center, cmap):
    x = IrrepTensor(x, len(x) // 2)
    x = SphericalTensor.from_irrep_tensor(x)

    return go.Surface(
        **x.plotly_surface(center=center, normalization='norm'),
        showscale=False,
        cmin=-0.33,
        cmax=0.33,
        colorscale=cmap,
    )


def main(args):
    if args.cmap == 'bwr':
        cmap = [[0, 'rgb(0,50,255)'], [0.5, 'rgb(200,200,200)'], [1, 'rgb(255,50,0)']]
    if args.cmap == 'plasma':
        cmap = [[0, '#9F1A9B'], [0.25, '#0D1286'], [0.5, '#000000'], [0.75, '#F58C45'], [1, '#F0F524']]

    if os.path.exists('gif'):
        shutil.rmtree('gif')
    os.makedirs('gif')

    for i, t in enumerate(tqdm(torch.linspace(0, 1, args.steps + 1)[:-1])):
        c1 = math.cos(2 * math.pi * t)
        s1 = math.sin(2 * math.pi * t)
        c2 = math.cos(4 * math.pi * t)
        s2 = math.sin(4 * math.pi * t)

        x1 = torch.tensor([0.1, 0.05, 0.2])
        x1 = 0.27 * x1 / x1.norm()

        x2 = c1 * torch.tensor([0.0, 0.0, 0.25, 0.0, 0.0])
        x2 += s1 * torch.tensor([0.25, 0.0, 0.0, 0., 0.0])
        x2 += s2 * torch.tensor([0.0, 0.25, 0.0, 0., 0.0])
        x2 += c2 * torch.tensor([0.0, 0., 0.0, 0., 0.25])
        x2 = 0.27 * x2 / x2.norm()

        x2 = torch.tensor([0.0, 0.0, 0.25, 0.0, 0.0])
        a = 2 * math.pi * t
        b = 2 * math.pi * t
        c = 2 * math.pi * t
        x1 = rs.rep(1, a, b, c) @ x1
        x2 = rs.rep(2, a, b, c) @ x2

        tp = rs.TensorProduct(1, 2, o3.selection_rule, normalization='norm')
        x3 = 4. * tp(x1, x2)
        out1, out2, out3 = rs.cut(x3, [1], [2], [3])

        data = [
            surf(x1, torch.tensor([-2, 0, 0.0]), cmap),
            surf(x2, torch.tensor([-1, 0, 0.0]), cmap),
            surf(out1, torch.tensor([0, 0, 0.0]), cmap),
            surf(out2, torch.tensor([1, 0, 0.0]), cmap),
            surf(out3, torch.tensor([2, 0, 0.0]), cmap),
        ]

        axis = dict(
            showbackground=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title='',
            nticks=3,
        )

        layout = dict(
            width=5 * args.height,
            height=args.height,
            scene=dict(
                xaxis=dict(
                    **axis,
                    range=[-2.5, 2.5]
                ),
                yaxis=dict(
                    **axis,
                    range=[-0.5, 0.5]
                ),
                zaxis=dict(
                    **axis,
                    range=[-0.5, 0.5]
                ),
                aspectmode='manual',
                aspectratio=dict(x=10, y=2, z=2),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=-2, z=0),
                    projection=dict(type='orthographic'),
                ),
                annotations=[
                    dict(
                        showarrow=False,
                        x=-1.5,
                        y=0,
                        z=0,
                        text=r"$\Huge\otimes$",
                        xanchor="center",
                        font=dict(color=args.color_text)
                    ),
                    dict(
                        showarrow=False,
                        x=-0.5,
                        y=0,
                        z=0,
                        text=r"$\Huge=$",
                        xanchor="center",
                        font=dict(color=args.color_text)
                    ),
                    dict(
                        showarrow=False,
                        x=0.5,
                        y=0,
                        z=0,
                        text=r"$\Huge\oplus$",
                        xanchor="center",
                        font=dict(color=args.color_text)
                    ),
                    dict(
                        showarrow=False,
                        x=1.5,
                        y=0,
                        z=0,
                        text=r"$\Huge\oplus$",
                        xanchor="center",
                        font=dict(color=args.color_text)
                    ),
                ]
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

    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--color_bg", type=str, default="rgba(0,0,0,0)")
    parser.add_argument("--color_text", type=str, default="rgb(255,255,255)")
    parser.add_argument("--cmap", type=str, default="bwr")

    args = parser.parse_args()

    main(args)
