# Cryptanalysis and Improved Attacks of Round-Reduced TEA and XTEA

This is the repository of the paper 'Cryptanalysis and Improved Attacks of Round-Reduced TEA and XTEA'. It contains the python codes to verify the practical attacks against round-reduced TEA and XTEA.

## Training neural distinguishers

Folder **saved_model** contains all our neural distinguishers of TEA and XTEA. File **train_teacher_distinguisher.py** contains the codes of searching for good difference constraint and training 8-round and 9-round neural distinguishers. And you can run **train_student_distinguisher.py** to train all the student neural distinguishers. File **util.py** contains the codes of testing the accuracy of every distinguisher, building lookup tables for the two 9-round distinguishers, and estimating the encryption speed of TEA and XTEA.

## Bit sensitivity Tests

Run **bit_sensitivity_test.py** to conduct the bit sensitivity test for every neural network. The results can be found in the folder **bit_sensitivity_res**.

## Key recovery attacks

The code of the 12-round key recovery attack on TEA is in **key_recovery_attack_tea.py** and the code of the 14-round key recovery attack on XTEA is in **key_recovery_attack_xtea.py**. We store the attack results in folder **attack_res**.
