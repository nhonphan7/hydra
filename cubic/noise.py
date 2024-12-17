import numpy as np
import pandas as pd


if __name__ == '__main__':
    random_state = 42
    noise_factor = 0.1
    
    np.random.seed(random_state)
    df = pd.read_csv('cubic.csv')
    psi = df['psi'].to_numpy()
    SD_voigt = df[['sd11', 'sd22', 'sd33', 'sd23', 'sd13', 'sd12']].to_numpy()

    for case in range(int(df['case'].max()) + 1):
        case_idx = df.index[df['case'] == case].to_list()

        Psi = psi[case_idx]
        SD11 = SD_voigt[case_idx, 0]
        SD22 = SD_voigt[case_idx, 1]
        SD33 = SD_voigt[case_idx, 2]
        SD23 = SD_voigt[case_idx, 3]
        SD13 = SD_voigt[case_idx, 4]
        SD12 = SD_voigt[case_idx, 5]
        
        Psi_range = Psi.max() - Psi.min()
        SD11_range = SD11.max() - SD11.min()
        SD22_range = SD22.max() - SD22.min()
        SD33_range = SD33.max() - SD33.min()
        SD23_range = SD23.max() - SD23.min()
        SD13_range = SD13.max() - SD13.min()
        SD12_range = SD12.max() - SD12.min()
        noise = noise_factor * np.random.uniform(
            low=-1., high=1., size=len(case_idx)
        )

        if Psi_range > 0:
            Psi += noise * Psi_range
        else:
            Psi += noise
        if SD11_range > 0:
            SD11 += noise * SD11_range
        else:
            SD11 += noise
        if SD22_range > 0:
            SD22 += noise * SD22_range
        else:
            SD22 += noise
        if SD33_range > 0:
            SD33 += noise * SD33_range
        else:
            SD33 += noise
        if SD23_range > 0:
            SD23 += noise * SD23_range
        else:
            SD23 += noise
        if SD13_range > 0:
            SD13 += noise * SD13_range
        else:
            SD13 += noise
        if SD12_range > 0:
            SD12 += noise * SD12_range
        else:
            SD12 += noise
        
        df.loc[case_idx, 'psi'] = Psi
        df.loc[case_idx, 'sd11'] = SD11
        df.loc[case_idx, 'sd22'] = SD22
        df.loc[case_idx, 'sd33'] = SD33
        df.loc[case_idx, 'sd23'] = SD23
        df.loc[case_idx, 'sd13'] = SD13
        df.loc[case_idx, 'sd12'] = SD12
    df.to_csv(f'cubic_noise{int(noise_factor * 100)}.csv', index=False)
