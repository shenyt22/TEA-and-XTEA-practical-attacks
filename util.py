from math import log2
import xtea
import tea
import numpy as np
from os import urandom
from keras.models import load_model
from time import time

word_size = xtea.WORD_SIZE()
block_size = xtea.BLOCK_SIZE()

# make a plaintext structure from a random plaintext pair
def make_plaintext_structure(diff, neutral_bits):
    p0l = np.frombuffer(urandom(4), dtype=np.uint32)
    p0r = np.frombuffer(urandom(4), dtype=np.uint32)
    for i in neutral_bits:
        if isinstance(i, int):
            i = [i]
        d0 = 0
        d1 = 0
        for j in i:
            if j < word_size:
                d1 |= (1 << j)
            else:
                d0 |= (1 << (j - word_size))
        p0l = np.concatenate([p0l, p0l ^ d0])
        p0r = np.concatenate([p0r, p0r ^ d1])
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r

# extract selected bits from an array of bit strings X which contains block_num blocks and then convert every sub-bit string to a integer
# return an array of integers
def extract_tar_bits(X, selected_bits, block_num=2):
    res = np.zeros(len(X), dtype=np.uint32)
    for i in range(len(selected_bits)):
        offset = block_num * len(selected_bits) - 1 - i
        index = block_size - 1 - selected_bits[i]
        for j in range(block_num):
            res = res | ((X[:, index]).astype(np.uint32) << offset)
            offset -= len(selected_bits)
            index += block_size
    return res

# extract selected bits from an array of bit strings X which contains 2 blocks
# return an array of sub-bit strings
def select_tar_bits(X, selected_bits):
    X_tmp = X.copy()
    index = [block_size - 1 - i for i in selected_bits] + [2 * block_size - 1 - i for i in selected_bits]
    index = np.array(index, dtype=np.uint32)
    return X_tmp[:, index]

def construct_lookup_table(net_path, selected_bits, saved_path):
    prior_input = np.arange(0, 2**(2*len(selected_bits)), 1, dtype=np.uint64)
    input = np.zeros((2**(2*len(selected_bits)), 2*len(selected_bits)), dtype=np.uint8)
    for i in range(len(selected_bits)):
        offset = len(selected_bits) + i
        index = len(selected_bits) - 1 - i
        input[:, index] = (prior_input >> offset) & 1
        offset -= len(selected_bits)
        index += len(selected_bits)
        input[:, index] = (prior_input >> offset) & 1
    net = load_model(net_path)
    Z = net.predict(input, batch_size=10000).flatten()
    Z = np.log2(Z / (1 - Z))
    np.save(saved_path, Z)


def test_distinguisher_acc(n, nr_range, net_path, diff, selected_bits=[i for i in range(block_size-1, -1, -1)], cipher=xtea):
    net = load_model(net_path)
    c0l, c0r, c1l, c1r, _ = cipher.make_target_diff_samples(n, nr_range, diff, True)
    X = cipher.convert_to_binary([c0l, c0r, c1l, c1r])
    X = select_tar_bits(X, selected_bits)
    Z = net.predict(X, batch_size=10000).flatten()
    tpr = np.sum(Z >= 0.5) / n

    c0l, c0r, c1l, c1r, _ = cipher.make_target_diff_samples(n, nr_range, diff, False)
    X = cipher.convert_to_binary([c0l, c0r, c1l, c1r])
    X = select_tar_bits(X, selected_bits)
    Z = net.predict(X, batch_size=10000).flatten()
    tnr = np.sum(Z < 0.5) / n

    print('tpr is {}, tnr is {}. acc is {}'.format(tpr, tnr, (tpr + tnr) / 2))

def test_distinguisher_acc_v2(n, nr_range, lookup_table_path, diff, selected_bits, cipher=tea):
    net = np.load(lookup_table_path)
    # positive samples
    c0l, c0r, c1l, c1r, _ = cipher.make_target_diff_samples(n, nr_range, diff, True)
    X = cipher.convert_to_binary([c0l, c0r, c1l, c1r])
    X = extract_tar_bits(X, selected_bits)
    Z = net[X]
    tpr = np.sum(Z >= 0) / n

    # negative samples
    c0l, c0r, c1l, c1r, _ = cipher.make_target_diff_samples(n, nr_range, diff, False)
    X = cipher.convert_to_binary([c0l, c0r, c1l, c1r])
    X = extract_tar_bits(X, selected_bits)
    Z = net[X]
    tnr = np.sum(Z < 0) / n

    print('tpr is {}, tnr is {}. acc is {}'.format(tpr, tnr, (tpr + tnr) / 2))

def test_encryption_time(n, batch_size, nr_range, encrypt):
    t = n // batch_size
    n = t * batch_size
    pl = np.frombuffer(urandom(4 * batch_size), dtype=np.uint32)
    pr = np.frombuffer(urandom(4 * batch_size), dtype=np.uint32)
    start = time()
    for _ in range(t):
        key = np.frombuffer(urandom(4 * 4), dtype=np.uint32).reshape(4, 1)
        cl, cr = encrypt((pl, pr), key, nr_range)
    end = time()
    print('time cost: {}s'.format(end - start))
    print('encryption process time:', log2(n / (end - start)))

if __name__ == '__main__':
    #-----------------------------------------construct-lookup-table-----------------------------------------
    # xtea
    nr = 9
    net_path = './saved_model/student/xtea/{}_distinguisher.h5'.format(nr)
    selected_bits = [38, 37, 33, 32, 1, 0]
    lookup_table = './lookup_table/xtea/0x80000000-0/38_37_33_32_1_0_9r.npy'
    construct_lookup_table(net_path=net_path, selected_bits=selected_bits, saved_path=lookup_table)

    # tea
    nr = 9
    net_path = './saved_model/student/tea/{}_distinguisher.h5'.format(nr)
    selected_bits = [38, 37, 33, 32, 1, 0]
    lookup_table = './lookup_table/tea/0x80000000-0/38_37_33_32_1_0_9r.npy'
    construct_lookup_table(net_path=net_path, selected_bits=selected_bits, saved_path=lookup_table)
    #-------------------------------------------------------------------------------------------------------

    #----------------------------------------test-distinguisher-accuracy-----------------------------------------
    diff = (0x80000000, 0)
    # xtea teacher net
    nr = 9
    nr_range = range(1, 1 + nr)
    net_path = './saved_model/xtea/0x80000000-0/{}_distinguisher.h5'.format(nr)
    test_distinguisher_acc(n=10**6, nr_range=nr_range, net_path=net_path, diff=diff, cipher=xtea)

    # xtea lookup table
    nr = 9
    nr_range = range(17, 17 + nr)
    lookup_table_path = './lookup_table/xtea/0x80000000-0/38_37_33_32_1_0_9r.npy'
    test_distinguisher_acc_v2(n=10**6, nr_range=nr_range, lookup_table_path=lookup_table_path, diff=diff, selected_bits=selected_bits, cipher=xtea)
    
    # xtea student net
    nr = 8
    nr_range = range(17, 17 + nr)
    net_path = './saved_model/student/xtea/{}_distinguisher.h5'.format(nr)
    selected_bits = [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
    test_distinguisher_acc(n=10**6, nr_range=nr_range, net_path=net_path, diff=diff, selected_bits=selected_bits, cipher=xtea)

    # tea teacher net
    nr = 9
    nr_range = range(1, 1 + nr)
    net_path = './saved_model/tea/0x80000000-0/{}_distinguisher.h5'.format(nr)
    test_distinguisher_acc(n=10**6, nr_range=nr_range, net_path=net_path, diff=diff, cipher=tea)

    # tea lookup table
    nr = 9
    nr_range = range(3, 3 + nr)
    selected_bits = [38, 37, 33, 32, 1, 0]
    lookup_table_path = './lookup_table/tea/0x80000000-0/38_37_33_32_1_0_9r.npy'
    test_distinguisher_acc_v2(n=10**6, nr_range=nr_range, lookup_table_path=lookup_table_path, diff=diff, selected_bits=selected_bits, cipher=tea)


    # tea student net
    nr = 8
    nr_range = range(3, 3 + nr)
    selected_bits_1 = [37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
    selected_bits_2 = [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
    net_path = './saved_model/student/tea/{}_distinguisher_1.h5'.format(nr)
    test_distinguisher_acc(n=10**7, nr_range=nr_range, net_path=net_path, diff=diff, selected_bits=selected_bits_1, cipher=tea)
    net_path = './saved_model/student/tea/{}_distinguisher_2.h5'.format(nr)
    test_distinguisher_acc(n=10**7, nr_range=nr_range, net_path=net_path, diff=diff, selected_bits=selected_bits_2, cipher=tea)
    #--------------------------------------------------------------------------------------------------------------

    #-----------------------------------------test-encryption-time-----------------------------------------
    # test tea encryption time
    test_encryption_time(n=2**30, batch_size=2**28, nr_range=(1, 13), encrypt=tea.encrypt)

    # test xtea encryption time
    test_encryption_time(n=2**30, batch_size=2**28, nr_range=(15, 29), encrypt=xtea.encrypt)
    #------------------------------------------------------------------------------------------------------