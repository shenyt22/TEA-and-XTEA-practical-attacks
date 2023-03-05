from cgitb import lookup
import tea
import numpy as np
from time import time
from os import urandom
from util import select_tar_bits, extract_tar_bits, make_plaintext_structure
from keras.models import load_model

word_size = tea.WORD_SIZE()

def gen_user_key():
    return np.frombuffer(urandom(4 * 4), dtype=np.uint32).reshape(4, 1)

def naive_attack_with_two_subkeys(c, key_guess_length, key_index, key_guess_round, lookup_table, selected_bits):
    c0l, c0r, c1l, c1r = c[0], c[1], c[2], c[3]
    n = len(c0l)
    total_key_guess_length = 2 * key_guess_length
    c0l = np.tile(c0l, 2**total_key_guess_length); c0r = np.tile(c0r, 2**total_key_guess_length); c1l = np.tile(c1l, 2**total_key_guess_length); c1r = np.tile(c1r, 2**total_key_guess_length)
    total_key_guess = np.arange(0, 2**total_key_guess_length, 1, dtype=np.uint64)
    key_guess_0 = (total_key_guess >> key_guess_length).astype(np.uint32)
    key_guess_1 = (total_key_guess & ((1 << key_guess_length) - 1)).astype(np.uint32)
    key_guess = [0, 0, 0, 0]
    key_guess[key_index[0]] = np.repeat(key_guess_0, n)
    key_guess[key_index[1]] = np.repeat(key_guess_1, n)
    d0l, d0r = tea.dec_one_round((c0l, c0r), key_guess, key_guess_round)
    d1l, d1r = tea.dec_one_round((c1l, c1r), key_guess, key_guess_round)
    X = tea.convert_to_binary([d0l, d0r, d1l, d1r])
    X = extract_tar_bits(X, selected_bits)
    Z = lookup_table[X]; Z = np.reshape(Z, (2**total_key_guess_length, n)); Z = np.sum(Z >= 0, axis=1)
    assert len(key_guess_0) == 2**total_key_guess_length
    if key_index[0] == 0:
        res = [(key_guess_0[i], key_guess_1[i], 0, 0) for i in range(2**total_key_guess_length) if Z[i] == n]
    else:
        res = [(0, 0, key_guess_0[i], key_guess_1[i]) for i in range(2**total_key_guess_length) if Z[i] == n]
    return res

def naive_attack_with_four_subkeys(c, key_guess_length, key_guess_offset, surviving_key, key_guess_rounds, net, selected_bits):
    c0l, c0r, c1l, c1r = c[0], c[1], c[2], c[3]
    n = len(c0l)
    res = []
    total_key_guess_length = 2 * key_guess_length[0] + 2 * key_guess_length[1]
    c0l = np.tile(c0l, 2**total_key_guess_length); c0r = np.tile(c0r, 2**total_key_guess_length); c1l = np.tile(c1l, 2**total_key_guess_length); c1r = np.tile(c1r, 2**total_key_guess_length)
    total_key_guess = np.arange(0, 2**total_key_guess_length, 1, dtype=np.uint64)
    mask_val_0 = (1 << key_guess_length[0]) - 1
    mask_val_1 = (1 << key_guess_length[1]) - 1
    key_guess_0_hi = (total_key_guess >> (total_key_guess_length - key_guess_length[0])).astype(np.uint32) << key_guess_offset[0]
    key_guess_1_hi = ((total_key_guess >> (2 * key_guess_length[1])) & mask_val_0).astype(np.uint32) << key_guess_offset[0]
    key_guess_2_hi = ((total_key_guess >> key_guess_length[1]) & mask_val_1).astype(np.uint32) << key_guess_offset[1]
    key_guess_3_hi = (total_key_guess & mask_val_1).astype(np.uint32) << key_guess_offset[1]
    for key_guess_0_low, key_guess_1_low, key_guess_2_low, key_guess_3_low in surviving_key:
        key_guess_0 = key_guess_0_hi | key_guess_0_low
        key_guess_1 = key_guess_1_hi | key_guess_1_low
        key_guess_2 = key_guess_2_hi | key_guess_2_low
        key_guess_3 = key_guess_3_hi | key_guess_3_low
        key_guess = [np.repeat(key_guess_0, n), np.repeat(key_guess_1, n), np.repeat(key_guess_2, n), np.repeat(key_guess_3, n)]
        d0l, d0r = tea.decrypt((c0l, c0r), key_guess, reversed(key_guess_rounds))
        d1l, d1r = tea.decrypt((c1l, c1r), key_guess, reversed(key_guess_rounds))
        X = tea.convert_to_binary([d0l, d0r, d1l, d1r])
        X = select_tar_bits(X, selected_bits)
        Z = net.predict(X, batch_size=10000); Z = np.reshape(Z, (2**total_key_guess_length, n)); Z = np.sum(Z >= 0.5, axis=1)
        assert len(key_guess_0) == 2**total_key_guess_length
        res += [(key_guess_0[i], key_guess_1[i], key_guess_2[i], key_guess_3[i]) for i in range(2**total_key_guess_length) if Z[i] == n]
    return res

def attack_with_three_steps(t, diff_1, diff_2, nr_range, key_guess_rounds, key_index_1, key_guess_length, lookup_table_path, net_paths, selected_bits, neutral_bits, max_sc=32, save_folder='./attack_res'):
    lookup_table = np.load(lookup_table_path)
    net_2, net_3 = load_model(net_paths[0]), load_model(net_paths[1])
    key_guess_offset = [[0, 0], [0, 0]]
    key_guess_offset[0][key_index_1[0] // 2] = key_guess_length[0]
    for i in range(2):
        key_guess_offset[1][i] = key_guess_offset[0][i] + key_guess_length[1][i]
    time_cost = np.zeros(t)
    success_attack = np.zeros(t, dtype=np.uint8)
    surviving_num = np.zeros(t, dtype=np.uint32)
    structure_consumption = np.zeros(t, dtype=np.uint32)
    for i in range(t):
        print('attack index:', i)
        sc = 0
        keys = gen_user_key()
        mask_vall = (1 << (key_guess_length[1][0] + key_guess_length[2][0])) - 1
        mask_valr = (1 << (key_guess_length[0] + key_guess_length[1][1] + key_guess_length[2][1])) - 1
        tk0, tk1 = keys[0][0] & mask_vall, keys[1][0] & mask_vall
        tk2, tk3 = keys[2][0] & mask_valr, keys[3][0] & mask_valr
        start = time()
        res = []
        while True:
            sc += 1
            if sc > max_sc:
                sc = max_sc
                break
            p0l, p0r, p1l, p1r = make_plaintext_structure(diff_1, neutral_bits)
            p2l, p2r = p0l ^ diff_2[0], p0r ^ diff_2[1]
            p3l, p3r = p1l ^ diff_2[0], p1r ^ diff_2[1]
            c0l, c0r = tea.encrypt((p0l, p0r), keys, nr_range)
            c1l, c1r = tea.encrypt((p1l, p1r), keys, nr_range)
            c2l, c2r = tea.encrypt((p2l, p2r), keys, nr_range)
            c3l, c3r = tea.encrypt((p3l, p3r), keys, nr_range)
            
            c = (c0l, c0r, c1l, c1r)
            surviving_key1 = naive_attack_with_two_subkeys(c, key_guess_length[0], key_index_1, key_guess_rounds[0], lookup_table, selected_bits[0])
            if len(surviving_key1) > 0:
                # print("surviving stage 1!")
                surviving_key2 = naive_attack_with_four_subkeys(c, key_guess_length[1], key_guess_offset[0], surviving_key1, key_guess_rounds[1], net_2, selected_bits[1])
                if len(surviving_key2) > 0:
                    res = naive_attack_with_four_subkeys(c, key_guess_length[2], key_guess_offset[1], surviving_key2, key_guess_rounds[2], net_3, selected_bits[2])
                    if len(res) > 0:
                        break
            
            c = (c0l, c0r, c2l, c2r)
            surviving_key1 = naive_attack_with_two_subkeys(c, key_guess_length[0], key_index_1, key_guess_rounds[0], lookup_table, selected_bits[0])
            if len(surviving_key1) > 0:
                # print("surviving stage 1!")
                surviving_key2 = naive_attack_with_four_subkeys(c, key_guess_length[1], key_guess_offset[0], surviving_key1, key_guess_rounds[1], net_2, selected_bits[1])
                if len(surviving_key2) > 0:
                    res = naive_attack_with_four_subkeys(c, key_guess_length[2], key_guess_offset[1], surviving_key2, key_guess_rounds[2], net_3, selected_bits[2])
                    if len(res) > 0:
                        break

            c = (c1l, c1r, c3l, c3r)
            surviving_key1 = naive_attack_with_two_subkeys(c, key_guess_length[0], key_index_1, key_guess_rounds[0], lookup_table, selected_bits[0])
            if len(surviving_key1) > 0:
                # print("surviving stage 1!")
                surviving_key2 = naive_attack_with_four_subkeys(c, key_guess_length[1], key_guess_offset[0], surviving_key1, key_guess_rounds[1], net_2, selected_bits[1])
                if len(surviving_key2) > 0:
                    res = naive_attack_with_four_subkeys(c, key_guess_length[2], key_guess_offset[1], surviving_key2, key_guess_rounds[2], net_3, selected_bits[2])
                    if len(res) > 0:
                        break

            c = (c2l, c2r, c3l, c3r)
            surviving_key1 = naive_attack_with_two_subkeys(c, key_guess_length[0], key_index_1, key_guess_rounds[0], lookup_table, selected_bits[0])
            if len(surviving_key1) > 0:
                # print("surviving stage 1!")
                surviving_key2 = naive_attack_with_four_subkeys(c, key_guess_length[1], key_guess_offset[0], surviving_key1, key_guess_rounds[1], net_2, selected_bits[1])
                if len(surviving_key2) > 0:
                    res = naive_attack_with_four_subkeys(c, key_guess_length[2], key_guess_offset[1], surviving_key2, key_guess_rounds[2], net_3, selected_bits[2])
                    if len(res) > 0:
                        break
        end = time()
        print('time cost: {}s'.format(end - start))
        time_cost[i] = end - start
        structure_consumption[i] = sc
        surviving_num[i] = len(res)
        print('structure consumption:', sc)
        if (tk0, tk1, tk2, tk3) in res:
            print('attack succeeds.')
            success_attack[i] = 1
            print('surviving key num:', len(res))
            for (k0, k1, k2, k3) in res:
                print('diff between key guess and true key: ({}, {}, {}, {})'.format(hex(tk0 ^ k0), hex(tk1 ^ k1), hex(tk2 ^ k2), hex(tk3 ^ k3)))
        elif len(res) > 0:
            print('attack fails.')
            success_attack[i] = 2
            print('surviving key num:', len(res))
            for (k0, k1, k2, k3) in res:
                print('diff between key guess and true key: ({}, {}, {}, {})'.format(hex(tk0 ^ k0), hex(tk1 ^ k1), hex(tk2 ^ k2), hex(tk3 ^ k3)))
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
    nr = 12
    nr_range = range(1, 1 + nr)
    key_guess_rounds = [nr, range(nr - 1, nr + 1), range(nr - 1, nr + 1)]
    key_index_1 = (2, 3)
    key_guess_length = [6, (5, 4), (5, 5)]
    lookup_table_path = './lookup_table/tea/0x80000000-0/38_37_33_32_1_0_9r.npy'
    net_paths = ['./saved_model/student/tea/8_distinguisher_1.h5', './saved_model/student/tea/8_distinguisher_2.h5']
    save_folder = './attack_res/tea/'
    selected_bits_1 = [38, 37, 33, 32, 1, 0]
    selected_bits_2 = [37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
    selected_bits_3 = [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
    selected_bits = [selected_bits_1, selected_bits_2, selected_bits_3]
    diff_1 = (0xc0200000, 0x84000000)
    diff_2 = (0x40200000, 0x84000000)
    neutral_bits = [32, 33, 34, 35, 36, 37]
    attack_with_three_steps(t=100, diff_1=diff_1, diff_2=diff_2, nr_range=nr_range, key_guess_rounds=key_guess_rounds, key_index_1=key_index_1, key_guess_length=key_guess_length, lookup_table_path=lookup_table_path,
                            net_paths=net_paths, selected_bits=selected_bits, neutral_bits=neutral_bits, max_sc=32, save_folder=save_folder)