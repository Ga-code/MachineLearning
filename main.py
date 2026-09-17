import numpy as np
import network
import pygame
import math
import time
import matplotlib.pyplot as plt
# pygame setup
pygame.init()
screen = pygame.display.set_mode((280, 280))
clock = pygame.time.Clock()
running = True
# data setup
"""
data = np.loadtxt('C:/Users/2gaut/Coding!/MachineLearning/mnist_train.csv', delimiter=',', skiprows=1)
test = np.loadtxt('C:/Users/2gaut/Coding!/MachineLearning/mnist_test.csv', delimiter=',', skiprows=1)
labels = data[:, 0].astype(int)
images = data[:, 1:]/255
labels_t = test[:, 0].astype(int)
images_t = test[:, 1:]/255
correct = np.hsplit(labels.T, 1875)
batches = np.hsplit(images.T, 1875)
correct_t = labels_t.T
batches_t = np.hsplit(images_t.T, 10000)
test1 = test[0, 1:].reshape(784, 1)/255
"""
weights = np.load('C:/Users/2gaut/Coding!/MachineLearning/weights.npz')
Ws = [weights['W0'], weights['W1'], weights['W2']]
bs = [weights['b0'], weights['b1'], weights['b2']]
MLP = network.MLP(Ws, bs)
#print(MLP.predict(test1))

#game loop
prev_pos = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            prev_pos = None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                surf = pygame.transform.smoothscale(screen, (28, 28))
                pixels = pygame.surfarray.array3d(surf)
                x = pixels[:, :, 0].T
                x = x/255
                
                nonzero = np.nonzero(x)
                min_row = min(nonzero[0])
                max_row = max(nonzero[0])
                min_col = min(nonzero[1])
                max_col = max(nonzero[1])
                cropped = x[min_row:max_row+1, min_col:max_col+1]
                cropped_u8 = (cropped*255).astype(np.uint8)
                surf = pygame.surfarray.make_surface(np.stack([cropped_u8.T]*3, axis=-1))
                h, w = cropped.shape
                if h > w:
                    w = int(20*w/h)
                    h = 20
                else:
                    h = int(20*h/w)
                    w = 20
                resized_surf  = pygame.transform.smoothscale(surf, (w, h))
                resized_array = pygame.surfarray.array3d(resized_surf)[:, :, 0]/255
                x = np.zeros((28, 28))
                start_row = (28 - h)//2
                start_col = (28 - w)//2
                x[start_row:start_row + h, start_col:start_col+w] = resized_array.T
                X = x.reshape(784, 1)
                print(MLP.predict(X))
                #plt.imshow(x, cmap='gray')
                #plt.show()
            if event.key == pygame.K_ESCAPE:
                running = False

    if pygame.mouse.get_pressed()[0]:
        curr_pos = np.array(pygame.mouse.get_pos())
        if prev_pos is not None:
            dist = curr_pos - prev_pos
            steps = np.linalg.norm(dist)/20
            steps = math.ceil(steps) + 3
            for i in range(0, int(steps)):
                pygame.draw.circle(screen, 'white', tuple(prev_pos + ((i+1)/steps)*dist), 10)
        else:
            pygame.draw.circle(screen, 'white', tuple(curr_pos), 10)
        prev_pos = curr_pos

    pygame.display.flip()
    clock.tick(60)

# training
"""
#epoch_n = 7
for i in range(0, epoch_n):
    for batch, c in zip(batches, correct):
        MLP.forward(batch)
        J, G_out = MLP.loss(c)
        MLP.backward(G_out)
        MLP.update(0.1)
        print(J)

Ws, bs, = MLP.save_weight()
np.savez('MachineLearning/weights.npz', W0=Ws[0], b0=bs[0], W1=Ws[1], b1=bs[1], W2=Ws[2], b2=bs[2])

# testing
num_correct = 0
for i in range(0, len(correct_t)):
   if MLP.predict(batches_t[i]) == correct_t[i]:
       num_correct+=1
print(f"Accuracy: {num_correct/len(correct_t)}")
"""

pygame.quit()