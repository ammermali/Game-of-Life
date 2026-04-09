import pygame

class Visualizer:
    def __init__(self, comm, params, cell_size=5):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.cell_size = cell_size
        self.width = params['width']
        self.height = params['height']

        if self.rank == 0:
            pygame.init()
            window_size = (self.width * self.cell_size, self.height * self.cell_size)
            self.screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Game of Life")
            self.clock = pygame.time.Clock()

    def draw(self, context):
        local_live_cells = []
        for agent in context.agents(1): # 1 è Cell.TYPE
            if agent.state == 1:
                local_live_cells.append((agent.point.x, agent.point.y))

        all_live_cells = self.comm.gather(local_live_cells, root=0)
        if self.rank == 0:
            self.screen.fill((255, 255, 255)) # sfondo bianco
            for rank_cells in all_live_cells:
                for x, y in rank_cells:
                    rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (0, 0, 240), rect) # Celle nere

            pygame.display.flip()
            self.clock.tick(15)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    import sys
                    sys.exit(0)