import numpy as np
from os import urandom

def WORD_SIZE():
    return 32

def BLOCK_SIZE():
    return 64

def delta():
    return 0x9e3779b9

def enc_one_round(p, k, round_num):
    L, R = p[0], p[1]
    i = (round_num + 1) // 2
    sum = np.uint32((i * delta()) & 0xffffffff)
    if round_num % 2 == 1:
        return (R, L + (((R << 4) + k[0]) ^ (R + sum) ^ ((R >> 5) + k[1])))
    else:
        return (R, L + (((R << 4) + k[2]) ^ (R + sum) ^ ((R >> 5) + k[3])))

def dec_one_round(c, k, round_num):
    L, R = c[0], c[1]
    i = (round_num + 1) // 2
    sum = np.uint32((i * delta()) & 0xffffffff)
    if round_num % 2 == 1:
        return (R - (((L << 4) + k[0]) ^ (L + sum) ^ ((L >> 5) + k[1])), L)
    else:
        return (R - (((L << 4) + k[2]) ^ (L + sum) ^ ((L >> 5) + k[3])), L)

def encrypt(p, k, nr_range):
    for i in nr_range:
        p = enc_one_round(p, k, i)
    return p

def decrypt(c, k, nr_range):
    for i in nr_range:
        c = dec_one_round(c, k, i)
    return c

def check_testvector():
    pz = np.zeros((1024, 1), dtype=np.uint32)
    for i in range(64):
        c = encrypt(pz[i : i + 2], pz[i + 2 : i + 6], range(1, 65))
        pz[i] = c[0]
        pz[i + 1] = c[1]
        pz[i + 6] = pz[i]
    if pz[63][0] == 0x2bb0f1b3 and pz[64][0] == 0xc023ed11 and pz[65][0] == 0x5c60bff2 and pz[66][0] == 0x7072d01c and pz[67][0] == 0x4513c5eb and pz[68][0] == 0x8f3a38ab:
        print('Testvector1 verified.')
    else:
        print('Testvector1 not verified.')
        return False
    
    n = 10**4
    p = np.frombuffer(urandom(8 * n), dtype=np.uint32).reshape(2, n)
    k = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, n)
    c = encrypt(p, k, range(1, 65))
    p_tmp = decrypt(c, k, reversed(range(1, 65)))
    if np.sum(p_tmp[0] == p[0]) == n and np.sum(p_tmp[1] == p[1]) == n:
        print('Testvector2 verified.')
    else:
        print('Testvector2 not verified.')
        return False
    
    return True

def convert_to_binary(arr):
    n = len(arr)
    X = np.zeros((n * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(n * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return(X)

def make_train_data(n, nr_range, diff):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    keys = np.frombuffer(urandom(4 * 4 * n), dtype=np.uint32).reshape(4, n)
    p0l = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    p0r = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    p1l = p0l ^ diff[0]; p1r = p0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    p1l[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
    p1r[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
    c0l, c0r = encrypt((p0l, p0r), keys, nr_range)
    c1l, c1r = encrypt((p1l, p1r), keys, nr_range)
    X = convert_to_binary([c0l, c0r, c1l, c1r])
    return (X, Y)

def make_target_diff_samples(n, nr_range, diff, positive_sample=True, given_key=None):
    p0l = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    p0r = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    if positive_sample:
        p1l = p0l ^ diff[0]
        p1r = p0r ^ diff[1]
    else:
        p1l = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        p1r = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    if given_key is None:
        keys = np.frombuffer(urandom(4 * 4 * n), dtype=np.uint32).reshape(4, n)
    else:
        keys = given_key
    c0l, c0r = encrypt((p0l, p0r), keys, nr_range)
    c1l, c1r = encrypt((p1l, p1r), keys, nr_range)
    return c0l, c0r, c1l, c1r, keys

if __name__ == '__main__':
    check_testvector()