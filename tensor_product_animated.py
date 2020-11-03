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


def get_cmap(x):
    if x == 'bwr':
        return [[0, 'rgb(0,50,255)'], [0.5, 'rgb(200,200,200)'], [1, 'rgb(255,50,0)']]
    if x == 'plasma':
        return [[0, '#9F1A9B'], [0.25, '#0D1286'], [0.5, '#000000'], [0.75, '#F58C45'], [1, '#F0F524']]


def surf(args, x, center):
    x = IrrepTensor(x, len(x) // 2)
    x = SphericalTensor.from_irrep_tensor(x)

    return go.Surface(
        **x.plotly_surface(args.res, center=center, normalization='norm'),
        showscale=False,
        cmin=-0.33,
        cmax=0.33,
        colorscale=get_cmap(args.cmap),
    )


def main(args):
    if args.out is None:
        p = {-1: 'o', 1: 'e'}
        args.out = f"{args.l1}{p[args.p1]}{args.l2}{p[args.p2]}"

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    Rs1 = [(1, args.l1, args.p1)]
    Rs2 = [(1, args.l2, args.p2)]
    tp = rs.TensorProduct(Rs1, Rs2, o3.selection_rule, normalization='norm')
    Rs_out = list(rs.split_by_mul(tp.Rs_out))

    x1 = rs.randn(Rs1)
    x1 = 0.27 * x1 / x1.norm()
    x2 = rs.randn(Rs2)
    x2 = 0.27 * x2 / x2.norm()

    for i, t in enumerate(tqdm(torch.linspace(0, 1, args.steps + 1)[:-1])):

        if args.animation == "rotation":
            a = 4 * math.pi * t
            b = 0
            c = 0
            p = 0 if t < 0.5 else 1
            gx1 = rs.rep(Rs1, a, b, c, p) @ x1
            gx2 = rs.rep(Rs2, a, b, c, p) @ x2
        if args.animation == "random":
            gx1 = rs.randn(Rs1)
            gx2 = rs.randn(Rs2)

        xleft = -0.5 * (len(Rs_out) + 2 - 1)
        data = [
            surf(args, gx1, torch.tensor([xleft, 0, 0.0])),
            surf(args, gx2, torch.tensor([xleft + 1, 0, 0.0])),
        ]

        outs = rs.cut(4. * tp(gx1, gx2), *Rs_out)
        data += [
            surf(args, out, torch.tensor([xleft + 2 + i, 0, 0.0]))
            for i, out in enumerate(outs)
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
            width=(2 + len(Rs_out)) * args.height,
            height=args.height,
            scene=dict(
                xaxis=dict(
                    **axis,
                    range=[xleft - 0.5, -xleft + 0.5]
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
                aspectratio=dict(x=2 * (2 + len(Rs_out)), y=2, z=2),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=-10, z=0),
                    projection=dict(type='orthographic'),
                ),
            ),
            paper_bgcolor=args.color_bg,
            plot_bgcolor=args.color_bg,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        fig = go.Figure(data=data, layout=layout)
        fig.write_image('{}/{:03d}.png'.format(args.out, i))

    subprocess.check_output(["convert", "-delay", "5", "-loop", "0", "-dispose", "2", "{}/*.png".format(args.out), "{}.gif".format(args.out)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--res", type=int, default=100)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--color_bg", type=str, default="rgba(0,0,0,0)")
    parser.add_argument("--color_text", type=str, default="rgb(255,255,255)")
    parser.add_argument("--cmap", type=str, default="plasma")
    parser.add_argument("--animation", type=str, default="rotation")
    parser.add_argument("--out", type=str)
    parser.add_argument("--l1", type=int, default=1)
    parser.add_argument("--l2", type=int, default=1)
    parser.add_argument("--p1", type=int, default=-1)
    parser.add_argument("--p2", type=int, default=-1)

    main(parser.parse_args())
