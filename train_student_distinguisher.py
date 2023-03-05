import student_net_xtea as sn_xtea
import student_net_tea as sn_tea

#-----------------------------------XTEA-----------------------------------
nr = 9
diff = (0x80000000, 0)
model_folder = './saved_model/student/xtea/'
selected_bits = [38, 37, 33, 32, 1, 0]
nr_range = range(17, 17 + nr)
sn_xtea.train_xtea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=diff, folder=model_folder, selected_bits=selected_bits)

nr = 8
selected_bits = [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
nr_range = range(17, 17 + nr)
sn_xtea.train_xtea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=diff, folder=model_folder, selected_bits=selected_bits)
#--------------------------------------------------------------------------

#-----------------------------------TEA------------------------------------
nr = 9
diff = (0x80000000, 0)
model_folder = './saved_model/student/tea/'
nr_range = range(3, 3 + nr)
selected_bits = [38, 37, 33, 32, 1, 0]
sn_tea.train_tea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=diff, folder=model_folder, selected_bits=selected_bits, net_suffix='')

nr = 8
nr_range = range(3, 3 + nr)
selected_bits_1 = [37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
sn_tea.train_tea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=diff, folder=model_folder, selected_bits=selected_bits_1, net_suffix='_1')
selected_bits_2 = [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 5, 4, 3, 2, 1, 0]
sn_tea.train_tea_distinguisher(num_epochs=10, nr_range=nr_range, depth=1, diff=diff, folder=model_folder, selected_bits=selected_bits_2, net_suffix='_2')
#--------------------------------------------------------------------------