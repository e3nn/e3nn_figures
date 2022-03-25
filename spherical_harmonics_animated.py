# pylint: disable=not-callable, no-member, invalid-name, missing-docstring, line-too-long
import argparse
import os
import shutil
import subprocess

import plotly.graph_objs as go
import torch
from tqdm.auto import tqdm
from e3nn.io import SphericalTensor
from e3nn import o3


def get_cmap(x):
    if x == 'bwr':
        return [[0, 'rgb(0,50,255)'], [0.5, 'rgb(200,200,200)'], [1, 'rgb(255,50,0)']]
    if x == 'plasma':
        return [[0, '#9F1A9B'], [0.25, '#0D1286'], [0.5, '#000000'], [0.75, '#F58C45'], [1, '#F0F524']]


def main(args):
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    x0 = torch.eye((args.L + 1)**2)
    st = SphericalTensor(args.L, p_val=1, p_arg=-1)

    centers = torch.tensor([
        [l - max(0, m), 0, l + min(0, m)]
        for l in range(args.L + 1)
        for m in range(-l, l + 1)
    ])

    centers = centers - torch.tensor([args.L / 2, 0, args.L / 2])
    centers = 2 * centers
    centers = centers * torch.tensor([1.0, 0.0, -1.0])

    for i, angle in enumerate(tqdm(torch.linspace(0, 2 * torch.pi, args.steps + 1)[:-1])):
        D = st.D_from_angles(*o3.axis_angle_to_angles(torch.tensor([1.0, 0.0, 1.0]), angle))
        x = torch.einsum("ij,zi->zj", x0, D)

        data = [
            go.Surface(
                **d,
                showscale=False,
                cmin=-0.33,
                cmax=0.33,
                colorscale=get_cmap(args.cmap),
            )
            for d in st.plotly_surface(x, res=args.res, centers=centers, normalization='norm')
        ]

        axis = dict(
            showbackground=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title='',
            nticks=3,
            range=[-args.L - 1.5, args.L + 1.5],
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
                aspectratio=dict(x=2, y=2, z=2),
                camera=dict(
                    up=dict(x=0, y=0, z=-1),
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
        fig.write_image('{}/{:03d}.png'.format(args.out, i))

    subprocess.check_output(["convert", "-delay", "3", "-loop", "0", "-dispose", "2", "{}/*.png".format(args.out), "{}.gif".format(args.out)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--res", type=int, default=100)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--color_bg", type=str, default="rgba(0,0,0,0)")
    parser.add_argument("--color_text", type=str, default="rgb(255,255,255)")
    parser.add_argument("--cmap", type=str, default="bwr")
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--out", type=str, default="gif")

    main(parser.parse_args())
