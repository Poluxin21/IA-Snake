import tensorflow as tf
from tensorflow import keras
import numpy as np
import pygame
import random
import time
sleep = time.sleep
# Modelo de treinamento da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu', use_bias=True),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
pygame.init()

# Inicia as variaveis
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
# define o número de amostras de treinamento
num_samples = 10000

# inicializa as matrizes de entrada e saída
x_train = np.zeros((num_samples, 4))
y_train = np.zeros((num_samples, 4))
score = 0
window_width = 1280
window_height = 720
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
PLAYER_SPEED = 300
directions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

dt = 0
font = pygame.font.SysFont("Arial", 28)
# Pega a posição do player
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() /  2)
# Update de diração para IA
def update_player_position(direction, dt):
    if direction == UP:
        player_pos.x -= PLAYER_SPEED * dt
    elif direction == DOWN:
        player_pos.y += PLAYER_SPEED * dt
    elif direction == LEFT:
        player_pos.x -= PLAYER_SPEED * dt
    elif direction == RIGHT:
        player_pos.y += PLAYER_SPEED * dt

# Desenha score
def update_score():
    font = pygame.font.Font(None, 36)
    text = font.render("Score: " + str(score), 1, (255, 255, 255))
    text_pos = text.get_rect(center=(screen.get_width()/2, 20))
    screen.blit(text, text_pos)


# Desenha comida
circle_pos = [random.randint(0, screen.get_width() - 10), random.randint(0, screen.get_height() - 10)]
def draw_circle(pos):
    pygame.draw.circle(screen, "green", pos, 10)
def update_circle():
    global circle_pos
    circle_pos = [random.randint(0, screen.get_width() - 10), random.randint(0, screen.get_height() - 10)]

# Detecta Colisões 
def detect_collision():
    global score
    distance = ((player_pos[0] - circle_pos[0]) ** 2 + (player_pos[1] - circle_pos[1]) ** 2) ** 0.5
    if distance < 10:
        score += 1
        update_score()
        update_circle()
    if player_pos[0] < 0 or player_pos[0] > screen.get_width() - 10:
        player_pos[0] = max(0, min(player_pos[0], screen.get_width() - 10))
    if player_pos[1] < 0 or player_pos[1] > screen.get_height() - 10:
        player_pos[1] = max(0, min(player_pos[1], screen.get_height() - 10))

# loop sobre o número de amostras de treinamento
for i in range(num_samples):
    # gera as posições aleatórias do jogador e da comida
    player_pos = [random.randint(0, window_width), random.randint(0, window_height)]
    circle_pos = [random.randint(0, window_width), random.randint(0, window_height)]

    # define a direção do jogador para chegar na comida
    if player_pos[0] < circle_pos[0]:
        x_train[i][0] = 1
    elif player_pos[0] > circle_pos[0]:
        x_train[i][1] = 1
    if player_pos[1] < circle_pos[1]:
        x_train[i][2] = 1
    elif player_pos[1] > circle_pos[1]:
        x_train[i][3] = 1

    # define a posição da comida
    y_train[i] = ((player_pos[0] - circle_pos[0]) ** 2 + (player_pos[1] - circle_pos[1]) ** 2) ** 0.5

direction = None
# Pega a posição do player
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() /  2)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # criar matriz x_test com as posições do jogador e da comida
    x_test = np.array([[player_pos.x, player_pos.y, circle_pos[0], circle_pos[1]]])

    # criar matriz y_test com as possíveis direções que o jogador pode tomar
    y_test = np.zeros((1, len(directions)))
    for i, dir in enumerate(directions):
        new_x = player_pos.x + dir[0] * 10
        new_y = player_pos.y + dir[1] * 10
        if new_x < 0 or new_x > window_width or new_y < 0 or new_y > window_height:
            # se a nova posição estiver fora da janela, definir valor como -1
            y_test[0, i] = -1
        else:
            # se a nova posição estiver dentro da janela, definir valor como 1
            y_test[0, i] = 1
    
    screen.fill("black")

    # Desenha o círculo da comida
    draw_circle(circle_pos)

    # Desenha o círculo do jogador
    player = pygame.draw.circle(screen, "red", player_pos, 10)

    # Faz o treinamento
    model.fit(x_train, y_train, epochs=1, batch_size=2, verbose=0)

    # Avalia o modelo
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Obtém as previsões da rede neural
    predictions = model.predict(x_train)

    # Obtém o índice da direção com a maior probabilidade
    direction_index = np.argmax(predictions)

    # Mapeia o índice para uma direção
    if direction_index == 0:
        direction = UP
        print("IA escolheu ir para cima")
    elif direction_index == 1:
        direction = DOWN
        print("IA escolheu ir para baixo")
    elif direction_index == 2:
        direction = LEFT
        print("IA escolheu ir para esquerda")
    elif direction_index == 3:
        direction = RIGHT
        print("IA escolheu ir para direita")
    if direction is not None:
        update_player_position(direction, dt)

    detect_collision()
    dt = clock.tick(60) / 1000
    pygame.display.flip()


pygame.quit()
