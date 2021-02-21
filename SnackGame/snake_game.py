"""
    2021/2/21 - now
    This a simply snack game and someday I will use AI techology to train it...
"""
import pygame
import random
from collections import namedtuple
from enum import Enum

pygame.init()

BLOCK_SIZE = 20
SPEED = 10

# rgb color block
WHITE = (225, 225, 225)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

font = pygame.font.SysFont("console", 25)

class Direction(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3

Point = namedtuple("Point", ["x", "y"])

class SnackGame:

    def __init__(self, width=640, height=480) -> None:
        self.w = width
        self.h = height

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snack")
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.Right

        # center of screen
        self.head = Point(self.w // 2, self.h // 2)
        self.snacks = [self.head, 
                       Point(self.head.x - BLOCK_SIZE, self.head.y), 
                       Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        # left top of point
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snacks:
            self._place_food()

    def play_step(self):
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.Left
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.Right
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.Down
                elif event.key == pygame.K_UP:
                    self.direction = Direction.Up
        # move
        self._move(self.direction)
        self.snacks.insert(0, self.head)
        
        # check if game is over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # place new food or just move

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snacks.pop()

        # update ui
        self._update_ui()
        self.clock.tick(SPEED)

        # return game is over or not and score
        game_over = False
        return game_over, self.score

    def _is_collision(self):
        # edge check
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y < 0 or self.head.y > self.h - BLOCK_SIZE:
            return True
        # body check
        if self.head in self.snacks[1:]:
            return True
        return False

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.Right:
            x += BLOCK_SIZE
        elif direction == Direction.Left:
            x -= BLOCK_SIZE
        elif direction == Direction.Up:
            y -= BLOCK_SIZE
        elif direction == Direction.Down:
            y += BLOCK_SIZE
        else:
            raise ValueError

        self.head = Point(x, y)

    def _update_ui(self):
        self.display.fill(BLACK)
        
        for point in self.snacks:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(point.x + 4, point.y + 4, BLOCK_SIZE // 2, BLOCK_SIZE // 2))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # text
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        

if __name__ == "__main__":
    game = SnackGame()

    # game loop
    while True:
        over, score = game.play_step()
        if over:
            break
        # break if game is over
    print(f"Finally Score is {score}")
    pygame.quit()