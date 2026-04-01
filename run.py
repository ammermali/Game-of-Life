from mpi4py import MPI
from src.Model import Model
from src.Cell import Cell
from repast4py import space

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    params = {
        'width': 100,
        'height': 100,
        'density': 0.5,
        'stop_at': 1000
    }

    model = Model(comm, params)
    model.run()