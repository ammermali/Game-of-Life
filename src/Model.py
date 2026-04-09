from repast4py import context, schedule, space
import numpy as np
from src.Cell import Cell
from src.UtilityCell import RationalCell
from src.Visualizer import Visualizer


class Model:
    def __init__(self, comm, params):
        self.comm = comm
        self.params = params
        self.context = context.SharedContext(comm=comm)
        if self.params.get('mode') == 'utility':
            self.cell_class = RationalCell
        else:
            self.cell_class = Cell

        box = space.BoundingBox(0, params['width'], 0, params['height'])
        self.grid = space.SharedGrid(
            name='grid',
            bounds=box,
            borders=space.BorderType.Sticky,
            occupancy=space.OccupancyType.Single,  # only one cell at a time
            buffer_size=1,
            comm=comm
        )

        self.context.add_projection(self.grid)
        self.initialize_agents(params)
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.visualizer = Visualizer(comm, params, cell_size=6)
        self.runner.schedule_repeating_event(1.1, 1, self.update_viz)
        self.runner.schedule_stop(params['stop_at'])

    def initialize_agents(self, params):
        rank = self.comm.Get_rank()
        local_bounds = self.grid.get_local_bounds()
        local_id = 0

        for i in range(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent):
            for j in range(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent):
                point = space.DiscretePoint(i, j)
                c = self.cell_class(local_id, rank, point)
                c.state = 1 if np.random.random() < params['density'] else 0
                self.context.add(c)
                self.grid.move(c, point)

                local_id += 1

    def step(self):
        self.context.synchronize(self.restore_cell)
        for agent in self.context.agents(Cell.TYPE):
            agent.set_next_state(self.grid)

        for agent in self.context.agents(Cell.TYPE):
            agent.apply_state()

    def restore_cell(self, data):
        uid, state, pt_coords = data
        cell = self.context.agent(uid)

        if cell is None:
            pt = space.DiscretePoint(*pt_coords)
            cell = self.cell_class(uid[0], uid[2], pt)

        cell.state = state
        return cell

    def update_viz(self):
        self.visualizer.draw(self.context)

    def run(self):
        self.runner.execute()