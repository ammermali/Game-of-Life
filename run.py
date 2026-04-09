import argparse
from mpi4py import MPI
from src.Model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['rule', 'utility'],
                        default='utility',
                        help='Choose between utility-based cells or rule-based cells.')

    parser.add_argument('--width', type=int, default=150, help='Width of the grid')
    parser.add_argument('--height', type=int, default=150, help='Height of the grid')
    parser.add_argument('--density', type=float, default=0.2, help='Initial density of live cells (0.0 to 1.0)')
    parser.add_argument('--stop_at', type=int, default=400, help='Tick to stop the simulation at')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    params = {
        'mode': args.mode,
        'width': args.width,
        'height': args.height,
        'density': args.density,
        'stop_at': args.stop_at
    }

    if comm.Get_rank() == 0:
        print(f"Mode: {params['mode'].upper()}")
        print(f"Grid: {params['width']}x{params['height']} (Density: {params['density']})")

    model = Model(comm, params)
    model.run()


if __name__ == '__main__':
    main()