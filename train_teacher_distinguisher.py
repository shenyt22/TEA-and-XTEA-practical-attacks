import teacher_net_xtea as tn_xtea
import teacher_net_tea as tn_tea
import xtea
import numpy as np

#-----------------------------------XTEA-----------------------------------
# find good difference constraint
nr = 8
nr_range = range(1, 1 + nr)
acc_record = np.zeros(xtea.BLOCK_SIZE())
model_folder = './saved_model/xtea/find_good_diff/'
tar_bits = [i for i in range(xtea.BLOCK_SIZE())]
for diff_bit in tar_bits:
    diff_l = 0
    diff_r = 0
    if diff_bit < xtea.WORD_SIZE():
        diff_r = 1 << diff_bit
    else:
        diff_l = 1 << (diff_bit - xtea.WORD_SIZE())
    test_folder = model_folder + '{}_'.format(diff_bit)
    _, h = tn_xtea.train_xtea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=(diff_l, diff_r), folder=test_folder)
    acc_record[diff_bit] = np.max(h.history['val_acc'])
    print('diff bit is', diff_bit)
    print('acc is', acc_record[diff_bit])
with open("./find_good_diff_res_xtea_{}r.txt".format(nr), "w") as f:
    for diff_bit in tar_bits:
        f.write('diff bit is {}, acc is {}\n'.format(diff_bit, acc_record[diff_bit]))

nr = 7
nr_range = range(1, 1 + nr)
diff = (0x80402010, 0)
model_folder = './saved_model/xtea/0x80402010-0/'
_, h = tn_xtea.train_xtea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=diff, folder=model_folder)
print('diff is 0x80402010/0, acc is', np.max(h.history['val_acc']))

# train distinguisher for difference (0x80000000, 0)
nr = 9
nr_range = range(1, 1 + nr)
diff = (0x80000000, 0)
model_folder = './saved_model/xtea/0x80000000-0/'
tn_xtea.train_xtea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=diff, folder=model_folder)
#--------------------------------------------------------------------------

#-----------------------------------TEA------------------------------------
# train distinguisher for difference (0x80000000, 0)
nr = 9
nr_range = range(1, 1 + nr)
diff = (0x80000000, 0)
model_folder = './saved_model/tea/0x80000000-0/'
tn_tea.train_tea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=diff, folder=model_folder)
#--------------------------------------------------------------------------