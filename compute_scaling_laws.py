from compute_sampling_ratio import compute_sampling_rates

SEQUENCE_SIZE = 128
BATCH_SIZE = 512
tokens, probabilities, sampling_rates = compute_sampling_rates(alpha=0.2)

total_tokens = {i: 0 for i in range(1, 11)}
total_tokens_once = 0
print(f'{"Dataset":<20}|\t{"Tokens":<20}|\t{"Probability":<20}|\t{"Sampling Rate":<20}|\t{"Multiplier (xN)":<20}|\t'
      f'{"1-epoch (Steps) CLM":<20}|\t{"1-epoch (Steps) MLM":<20}|\t{"1-epoch over ECtHR (Tokens)":<20}')
print('-' * 200)
for key in probabilities.keys():
    multiplier = f'{sampling_rates[key]/sampling_rates["ecthr-cases"]:.1f}x'
    probability = f'{probabilities[key]*100:.1f}%'
    local_tokens = f'{tokens[key]/1e6:.1f}M'
    sampling_rate = f'{sampling_rates[key]*100:.1f}%'
    all_ecthr = f'{float(multiplier[:-1])*78.5:.1f}M'
    all_steps = f'{(float(all_ecthr[:-1])*1000)/(SEQUENCE_SIZE*BATCH_SIZE):.1f}K'
    all_steps_mlm = f'{(float(all_steps[:-1])*(1/0.15)):.1f}K'
    total_tokens_once += float(all_ecthr[:-1])
    all_steps_ = f'{float(all_ecthr[:-1]) / (SEQUENCE_SIZE * BATCH_SIZE):.3f}M'
    for k in total_tokens.keys():
        total_tokens[k] += float(multiplier[:-1]) * k * 78.5
    print(f'{key:<20}|\t{local_tokens:>20}|\t{probability:>20}|\t{sampling_rate:>20}|'
          f'\t{multiplier:>20}|\t{all_steps:>20}|\t{all_steps_mlm:>20}|\t{all_ecthr:>20}')

setting_tuples = []
for key, value in total_tokens.items():
    setting_tuples.append((value/1000, value/20, value/SEQUENCE_SIZE, value / (SEQUENCE_SIZE * BATCH_SIZE), (value * (1/0.15)) / (SEQUENCE_SIZE * BATCH_SIZE)))


print('-' * 200)
print(F'{"Statistics per N epochs":>20}\t\t\t' + '\t'.join([f'     {n}  |' for n in range(10)]))
print('-' * 200)
print(f'Total number of tokens:\t\t\t' + '\t'.join([f'  {value[0]:.1f}B |' for value in setting_tuples]))
print(f'Total number of parameters:\t\t' + '\t'.join([f'{value[1]:.1f}M |' for value in setting_tuples]))
print(f'Total number of sequences\t\t' + '\t'.join([f' {value[2]:.1f}M |' for value in setting_tuples]))
print(f'Total number of CLM steps:\t\t' + '\t'.join([f' {value[3]:.2f}M |' for value in setting_tuples]))
print(f'Total number of MLM steps:\t\t' + '\t'.join([f' {value[4]:.2f}M |' for value in setting_tuples]))

'''
Dataset             |	Probability         |	Sampling Rate       |	Multiplier (xN)     |	1-epoch (Steps) CLM |	1-epoch (Steps) MLM |	1-epoch over ECtHR (Tokens)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
eu-legislation      |	                1.3%|	                5.5%|	                1.7x|	                1.6K|	               10.7K|	              106.9M
eu-court-cases      |	                1.0%|	                4.8%|	                1.5x|	                1.4K|	                9.3K|	               94.3M
uk-legislation      |	                0.8%|	                4.3%|	                1.4x|	                1.3K|	                8.7K|	               88.1M
uk-court-cases      |	                2.0%|	                6.9%|	                2.2x|	                2.1K|	               14.0K|	              138.4M
indian-court-cases  |	                0.6%|	                3.8%|	                1.2x|	                1.2K|	                8.0K|	               75.5M
us-contracts        |	               29.0%|	               26.2%|	                8.2x|	                7.9K|	               52.7K|	              515.8M
us-court-cases      |	               62.9%|	               38.6%|	               12.1x|	               11.6K|	               77.3K|	              761.1M
legal-c4            |	                1.9%|	                6.7%|	                2.1x|	                2.0K|	               13.3K|	              132.1M
ecthr-cases         |	                0.4%|	                3.2%|	                1.0x|	                1.0K|	                6.7K|	               62.9M
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Statistics per N epochs			     0  |	     1  |	     2  |	     3  |	     4  |	     5  |	     6  |	     7  |	     8  |	     9  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total number of tokens:			  2.0B |	  4.0B |	  5.9B |	  7.9B |	  9.9B |	  11.9B |	  13.8B |	  15.8B |	  17.8B |	  19.8B |
Total number of parameters:		 98.8M |	197.5M |	296.3M |	395.0M |	493.8M |	592.5M |	691.3M |	790.0M |	888.8M |	987.5M |
Total number of sequences		 15.4M |	 30.9M |	 46.3M |	 61.7M |	 77.2M |	 92.6M |	 108.0M |	 123.4M |	 138.9M |	 154.3M |
Total number of CLM steps:		 0.03M |	 0.06M |	 0.09M |	 0.12M |	 0.15M |	 0.18M |	 0.21M |	 0.24M |	 0.27M |	 0.30M |
Total number of MLM steps:		 0.20M |	 0.40M |	 0.60M |	 0.80M |	 1.00M |	 1.21M |	 1.41M |	 1.61M |	 1.81M |	 2.01M |
'''
