import xtea
import tea
import numpy as np
import util
from keras.models import load_model
from os import urandom
import copy

# type = 0, cal sen_j
# type = 1, cal sen'_j
def make_target_bit_diffusion_data(X, id=0, type=0, block_size=64):
    n = X.shape[0]
    masks = np.frombuffer(urandom(n), dtype=np.uint8) & 0x1
    masked_X = copy.deepcopy(X)
    if type == 0:
        masked_X[:, block_size - 1 - id] = X[:, block_size - 1 - id] ^ masks
    else:
        masked_X[:, block_size - 1 - id] = X[:, block_size - 1 - id] ^ masks
        masked_X[:, block_size * 2 - 1 - id] = X[:, block_size * 2 - 1 - id] ^ masks
    return masked_X

def test_bits_sensitivity(n, nr_range, net_path, diff=(0x80000000, 0), folder='./bits_sensitivity_res/', cipher=xtea):
    block_size = cipher.BLOCK_SIZE()
    acc = np.zeros(block_size + 1)
    X, Y = cipher.make_train_data(n=n, nr_range=nr_range, diff=diff)
    net = load_model(net_path)
    _, acc[block_size] = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is', acc[block_size])

    for i in range(block_size):
        masked_X = make_target_bit_diffusion_data(X, id=i, type=0, block_size=block_size)
        # masked_X = make_target_bit_diffusion_data(X, id=i, type=1, block_size=block_size)
        _, acc[i] = net.evaluate(masked_X, Y, batch_size=10000, verbose=0)
        print('cur bit position is', i)
        print('the decrease of the acc is', acc[block_size] - acc[i])
    
    np.save(folder + '{}_distinguisher_bit_sensitivity.npy'.format(nr), acc)

def test_bits_sensitivity_v2(n, nr_range, net_path, diff, selected_bits, cipher=tea):
    block_size = len(selected_bits)
    acc = np.zeros(block_size + 1)
    X, Y = cipher.make_train_data(n=n, nr_range=nr_range, diff=diff)
    X = util.select_tar_bits(X, selected_bits)
    net = load_model(net_path)
    _, acc[block_size] = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is', acc[block_size])

    for i in range(block_size):
        # masked_X = make_target_bit_diffusion_data(X, id=i, type=0, block_size=block_size)
        masked_X = make_target_bit_diffusion_data(X, id=i, type=1, block_size=block_size)
        _, acc[i] = net.evaluate(masked_X, Y, batch_size=10000, verbose=0)
        print('cur bit position is', i)
        print('the decrease of the acc is', acc[block_size] - acc[i])

# -----------------------------------XTEA-----------------------------------
diff = (0x80000000, 0)

# test bit sensitivity for teacher net
nr = 8
nr_range = range(1, 1 + nr)
net_path = './saved_model/xtea/0x80000000-0/{}_distinguisher.h5'.format(nr)
folder = './bit_sensitivity_res/xtea/0x80000000-0/'
test_bits_sensitivity(n=10**6, nr_range=nr_range, net_path=net_path, diff=diff, folder=folder, cipher=xtea)

# test bit sensitivity for student net
nr = 8
nr_range = range(17, 17 + nr)
net_path = './saved_model/student/xtea/{}_distinguisher.h5'.format(net_path)
selected_bits = [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
test_bits_sensitivity_v2(n=10**6, nr_range=nr_range, net_path=net_path, diff=diff, selected_bits=selected_bits, cipher=xtea)
# --------------------------------------------------------------------------

# -----------------------------------TEA------------------------------------
# test bit sensitivity for teacher net
nr = 9
nr_range = range(1, 1 + nr)
net_path = './saved_model/tea/0x80000000-0/{}_distinguisher.h5'.format(nr)
folder = './bit_sensitivity_res/tea/0x80000000-0/'
test_bits_sensitivity(n=10**6, nr_range=nr_range, net_path=net_path, diff=diff, folder=folder, cipher=tea)

# test bit sensitivity for student net
nr = 8
nr_range = range(3, 3 + nr)
selected_bits_1 = [37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
selected_bits_2 = [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
net_path = './saved_model/student/tea/{}_distinguisher_1.h5'.format(nr)
test_bits_sensitivity_v2(n=10**6, nr_range=nr_range, net_path=net_path, diff=diff, selected_bits=selected_bits_1, cipher=tea)
net_path = './saved_model/student/tea/{}_distinguisher_2.h5'.format(nr)
test_bits_sensitivity_v2(n=10**6, nr_range=nr_range, net_path=net_path, diff=diff, selected_bits=selected_bits_2, cipher=tea)
# --------------------------------------------------------------------------