import xtea
import numpy as np
import util
from time import time
from os import urandom
from keras.models import load_model

word_size = xtea.WORD_SIZE()

def gen_user_key():
    return np.frombuffer(urandom(4 * 4), dtype=np.uint32).reshape(4, 1)

# guess one subkeys k1[15~0]
def attack_in_stage1(c, key_guess_length, key_index, key_guess_rounds, lookup_table, selected_bits):
    c0l, c0r, c1l, c1r = c[0], c[1], c[2], c[3]
    n = len(c0l)
    c0l = np.tile(c0l, 2**key_guess_length); c0r = np.tile(c0r, 2**key_guess_length); c1l = np.tile(c1l, 2**key_guess_length); c1r = np.tile(c1r, 2**key_guess_length)
    tar_key_guess = np.arange(0,2**key_guess_length, 1, dtype=np.uint32)
    tar_key_guess = np.repeat(tar_key_guess, n)
    key_guess = [0, 0, 0, 0]
    key_guess[key_index] = tar_key_guess
    d0l, d0r = xtea.decrypt((c0l, c0r), key_guess, reversed(key_guess_rounds))
    d1l, d1r = xtea.decrypt((c1l, c1r), key_guess, reversed(key_guess_rounds))
    X = xtea.convert_to_binary([d0l, d0r, d1l, d1r])
    X = util.extract_tar_bits(X, selected_bits)
    Z = lookup_table[X]; Z = np.reshape(Z, (2**key_guess_length, n)); Z = np.sum(Z >= 0, axis=1)
    res = [i for i in range(2**key_guess_length) if Z[i] == n]
    return res

# guess two subkeys (k0[9~0], k1[24~16])
def attack_in_stage2(c, key_guess_length1, key_guess_offset1, key_guess_length2, key_index1, key_index2, key_guess_rounds, net, selected_bits, survived_keys):
    res = []
    c0l, c0r, c1l, c1r = c[0], c[1], c[2], c[3]
    n = len(c0l)
    key_guess_length = key_guess_length1 + key_guess_length2
    c0l = np.tile(c0l, 2**key_guess_length); c0r = np.tile(c0r, 2**key_guess_length); c1l = np.tile(c1l, 2**key_guess_length); c1r = np.tile(c1r, 2**key_guess_length)
    key_guess = np.arange(0, 2**key_guess_length, 1, dtype=np.uint32)
    key_guess = np.repeat(key_guess, n)
    key_guess_1_hi = key_guess & ((1 << key_guess_length1) - 1)
    key_guess_1_hi = key_guess_1_hi << key_guess_offset1
    key_guess_2 = key_guess >> key_guess_length1
    for key_guess_1_low in survived_keys:
        key_guess_1 = key_guess_1_hi | key_guess_1_low
        key_guess = [0, 0, 0, 0]
        key_guess[key_index1] = key_guess_1
        key_guess[key_index2] = key_guess_2
        d0l, d0r = xtea.decrypt((c0l, c0r), key_guess, reversed(key_guess_rounds))
        d1l, d1r = xtea.decrypt((c1l, c1r), key_guess, reversed(key_guess_rounds))
        X = xtea.convert_to_binary([d0l, d0r, d1l, d1r])
        X = util.select_tar_bits(X, selected_bits)
        Z = net.predict(X, batch_size=10000).flatten()
        Z = np.reshape(Z, (2**key_guess_length, n)); Z = np.sum(Z >= 0.5, axis=1)
        res += [(key_guess_1[i*n], key_guess_2[i*n]) for i in range(len(Z)) if Z[i] == n]
    return res

# guess two subkeys (k0[9~0], k1[24~0]) in two steps
def attack_with_structure_two_steps(t, diff_1, diff_2, nr_range, key_guess_rounds, key_index, key_guess_length, lookup_table_path, net_path, selected_bits, neutral_bits, max_sc=16, save_folder='./attack_res/xtea/'):
    lookup_table = np.load(lookup_table_path)
    net = load_model(net_path)
    time_cost = np.zeros(t)
    success_attack = np.zeros(t, dtype=np.uint8)
    surviving_num = np.zeros(t, dtype=np.uint32)
    structure_consumption = np.zeros(t, dtype=np.uint32)
    for i in range(t):
        print('attack index:', i)
        sc = 0
        keys = gen_user_key()
        mask_val1 = (1 << (key_guess_length[0] + key_guess_length[1][0])) - 1
        mask_val2 = (1 << key_guess_length[1][1]) - 1
        tk1 = keys[key_index[0]][0] & mask_val1
        tk2 = keys[key_index[1]][0] & mask_val2
        start = time()
        res = []
        while True:
            sc += 1
            if sc > max_sc:
                sc = max_sc
                break
            p0l, p0r, p1l, p1r = util.make_plaintext_structure(diff_1, neutral_bits)
            p2l, p2r = p0l ^ diff_2[0], p0r ^ diff_2[1]
            p3l, p3r = p1l ^ diff_2[0], p1r ^ diff_2[1]
            c0l, c0r = xtea.encrypt((p0l, p0r), keys, nr_range)
            c1l, c1r = xtea.encrypt((p1l, p1r), keys, nr_range)
            c2l, c2r = xtea.encrypt((p2l, p2r), keys, nr_range)
            c3l, c3r = xtea.encrypt((p3l, p3r), keys, nr_range)

            surviving_key = attack_in_stage1((c0l, c0r, c1l, c1r), key_guess_length[0], key_index[0], key_guess_rounds[0], lookup_table, selected_bits[0])
            if len(surviving_key) > 0:
                res = attack_in_stage2((c0l, c0r, c1l, c1r), key_guess_length[1][0], key_guess_length[0], key_guess_length[1][1], key_index[0], key_index[1], key_guess_rounds[1], net, selected_bits[1], surviving_key)
                if len(res) > 0:
                    break

            surviving_key = attack_in_stage1((c0l, c0r, c2l, c2r), key_guess_length[0], key_index[0], key_guess_rounds[0], lookup_table, selected_bits[0])
            if len(surviving_key) > 0:
                res = attack_in_stage2((c0l, c0r, c2l, c2r), key_guess_length[1][0], key_guess_length[0], key_guess_length[1][1], key_index[0], key_index[1], key_guess_rounds[1], net, selected_bits[1], surviving_key)
                if len(res) > 0:
                    break

            surviving_key = attack_in_stage1((c1l, c1r, c3l, c3r), key_guess_length[0], key_index[0], key_guess_rounds[0], lookup_table, selected_bits[0])
            if len(surviving_key) > 0:
                res = attack_in_stage2((c1l, c1r, c3l, c3r), key_guess_length[1][0], key_guess_length[0], key_guess_length[1][1], key_index[0], key_index[1], key_guess_rounds[1], net, selected_bits[1], surviving_key)
                if len(res) > 0:
                    break

            surviving_key = attack_in_stage1((c2l, c2r, c3l, c3r), key_guess_length[0], key_index[0], key_guess_rounds[0], lookup_table, selected_bits[0])
            if len(surviving_key) > 0:
                res = attack_in_stage2((c2l, c2r, c3l, c3r), key_guess_length[1][0], key_guess_length[0], key_guess_length[1][1], key_index[0], key_index[1], key_guess_rounds[1], net, selected_bits[1], surviving_key)
                if len(res) > 0:
                    break
        end = time()
        print('time cost: {}s'.format(end - start))
        time_cost[i] = end - start
        structure_consumption[i] = sc
        surviving_num[i] = len(res)
        print('structure consumption:', sc)
        if (tk1, tk2) in res:
            print('attack succeeds.')
            success_attack[i] = 1
            print('surviving key num:', len(res))
            for (k1, k2) in res:
                print('diff between key guess and true key: ({}, {})'.format(hex(tk1 ^ k1), hex(tk2 ^ k2)))
        elif len(res) > 0:
            print('attack fails.')
            success_attack[i] = 2
            print('surviving key num:', len(res))
            for (k1, k2) in res:
                print('diff between key guess and true key: ({}, {})'.format(hex(tk1 ^ k1), hex(tk2 ^ k2)))
        else:
            success_attack[i] = 0
            print('attack fails.')
        
    print('average time cost: {}s'.format(np.mean(time_cost)))
    print('average surviving num with tk surviving', np.mean(surviving_num[success_attack == 1]))
    print('average surviving num with key surviving', np.mean(surviving_num[success_attack > 0]))
    print('success rate:', np.sum(success_attack == 1) / t)
    print('fake key surviving rate:', np.sum(success_attack == 2) / t)
    print('average structure consumption:', np.mean(structure_consumption))

    np.save(save_folder+'success_attack.npy', success_attack)
    np.save(save_folder+'structure_consumption.npy', structure_consumption)
    np.save(save_folder+'time_cost.npy', time_cost)
    np.save(save_folder+'surviving_num.npy', surviving_num)


if __name__ == '__main__':
    # 2 + 8 + 4, guess two subkeys
    nr_range = range(15, 29)
    key_guess_rounds = [range(26, 29), range(25, 29)]
    key_guess_length = [16, (9, 10)]
    key_index = [1, 0]
    lookup_table_path = './lookup_table/xtea/0x80000000-0/38_37_33_32_1_0_9r.npy'
    net_path = './saved_model/student/xtea/8_distinguisher.h5'
    selected_bits = [[38, 37, 33, 32, 1, 0], [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]]
    neutral_bits = [32, 33, 34, 35, 36]
    attack_with_structure_two_steps(t=1000, diff_1=(0xc0200000, 0x84000000), diff_2=(0x40200000, 0x84000000), nr_range=nr_range, key_guess_rounds=key_guess_rounds, key_index=key_index, key_guess_length=key_guess_length, 
                                    lookup_table_path=lookup_table_path, net_path=net_path, selected_bits=selected_bits, neutral_bits=neutral_bits, max_sc=16)