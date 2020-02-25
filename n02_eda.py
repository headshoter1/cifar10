import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from n01_config import get_paths
#датасет можно скачать тут https://www.kaggle.com/c/cifar-10/data
DATA = get_paths()['dataset']
# LABELS = ['CZ_N01_0a00000', 'RU_N02_a000aa100', 'VN_N03_00a0_0000', 'GB_N06_a00aaa', 'KG_N15_a00000', 'AM_N04_00aa000', 'NL_N02_0aaa00', 'KG_N02_a0000aa', 'RU_N03_a000aa700', 'LU_N01_aa0000', 'KW_N01_0000000', 'GB_N08_a0aaa', 'AE_N02_a0000', 'AE_N06_000000', 'RO_N02_aa00aaa', 'KG_N11_00000_aa', 'PL_N07_aaa0a00', 'KG_N18_0000_aa', 'PT_N04_a000000', 'KZ_N01_000aaa00', 'FR_N05_aaa000aa', 'AE_N07_0000000', 'GE_N02_aa_000aa', 'KZ_N11_000aa00', 'KG_N19_aaaa0000', 'FR_N02_0000aa00', 'HK_N04_aa_00', 'KZ_N20_0000_aa', 'KG_N16_0000_aaaa', 'KG_N05_0000aa', 'ES_N04_aa0000a', 'ES_N09_0000_aaa', 'ES_N02_a0000aa', 'GE_N03_aaa000', 'HK_N02_aa_0000', 'AM_N01_000aa00', 'FR_N01_aa000aa', 'GB_N01_aa00aaa', 'RU_N05_aa000000', 'KZ_N05_a000aa', 'KZ_N03_a000aaa', 'AM_N03_00aa000', 'ES_N06_aa0000aa', 'KR_N03_00a0000', 'AE_N05_a0000', 'KR_N01_00a0000', 'NL_N03_aa000a', 'BY_N12_aa0000', 'RU_N06_aa00000', 'ES_N01_0000aaa', 'PT_N05_00_aa00', 'HK_N03_aa_000', 'SE_N01_aaa000', 'HK_N01_aa0000', 'FR_N03_000aa00', 'PT_N03_00aa00', 'KG_N03_0000_aaa', 'BE_N02_0aaa000', 'KZ_N07_a000_aaa', 'GB_N05_a000aaa', 'LU_N04_aa_0000', 'ES_N03_a0000aa', 'NL_N01_00aaa0', 'BG_N02_a0000aa', 'KW_N05_00_00000', 'UZ_N05_00a000aa', 'ES_N08_a0000aaa', 'AZ_N03_00aa000', 'PT_N01_0000aa', 'AM_N02_000aa00', 'KZ_N09_a000aa', 'KG_N06_000aa', 'BY_N02_0aa0000', 'AE_N10_00_00000', 'ME_N01_aaaa000', 'FR_N04_aa_000aa', 'AZ_N02_00_aa000', 'VN_N04_00a0_00000', 'GE_N01_aa000aa', 'RU_N01_a000aa00', 'KG_N09_00000aaa', 'VN_N02_00a_00000', 'KZ_N06_a000_aa', 'FR_N05_000aaa00', 'GE_N04_aaa_000', 'KZ_N02_00_000_aaa', 'KG_N04_0000_aa', 'ME_N02_aa_aa000', 'LU_N02_00000', 'NL_N14_aa0000', 'KG_N10_00000aa', 'KG_N17_00000_aaa', 'NL_N04_00aaaa', 'PL_N01_aa0a000', 'LU_N03_0000', 'KG_N01_a0000a', 'LV_N05_aa00', 'AE_N01_a00000', 'AE_N09_0_00000', 'MD_N03_aaa000', 'SE_N02_aaa000', 'ME_N03_aa_aa00', 'NL_N05_aaaa00', 'KZ_N13_a000000', 'AE_N04_a00000', 'PT_N06_00_00aa', 'KZ_N04_000aa00', 'ES_N07_aa0000aa', 'AZ_N04_00_a000', 'KG_N13_000000a', 'BG_N01_aa0000aa', 'KR_N02_00a0000', 'PT_N02_aa0000', 'MD_N01_aaaa000', 'AZ_N01_00aa000', 'VN_N01_00a00000', 'ES_N05_aa0000a']
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    df = pd.read_csv(os.path.join(DATA['path'], DATA['lables_csv']))
    print(df.head())

    labels = sorted(set(df['label'].tolist()))
    print(labels)

    for label in LABELS:
        tdf = df[df['label'] == label]
        print(f'{label}: {tdf.shape[0]}')

    label = LABELS[0]
    tdf = df[df['label'] == label]
    tdf.reset_index(inplace=True, drop=True)

    plt.figure()
    for i in range(9):
        plt.subplot(3,3,1+i)
        filename = f'{tdf.loc[i, "id"]}.png'
        img = cv2.imread(os.path.join(DATA['path'], DATA['train_dir'], filename))[:,:,::-1]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        plt.imshow(img)

    plt.show()


if __name__ == '__main__':
    main()
