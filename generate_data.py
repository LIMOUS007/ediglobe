import pandas as pd
import numpy as np

np.random.seed(42)

num_rows = 1338

age = np.random.randint(18, 65, num_rows)
sex = np.random.choice(['female', 'male'], num_rows)
bmi = np.random.uniform(18.0, 45.0, num_rows)
children = np.random.randint(0, 6, num_rows).astype(float)
smoker = np.random.choice(['yes', 'no'], num_rows)
claim_amount = np.random.uniform(500.0, 50000.0, num_rows)
past_consultations = np.random.randint(0, 10, num_rows).astype(float)
num_of_steps = np.random.randint(2000, 15000, num_rows).astype(float)
hospital_expenditure = np.random.uniform(1000.0, 100000.0, num_rows)
number_of_past_hospitalizations = np.random.randint(0, 5, num_rows).astype(float)
anual_salary = np.random.uniform(30000.0, 200000.0, num_rows)
region = np.random.choice(['southwest', 'southeast', 'northeast', 'northwest'], num_rows)
charges = np.random.uniform(1000.0, 60000.0, num_rows)

data = {
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'Claim_Amount': claim_amount,
    'past_consultations': past_consultations,
    'num_of_steps': num_of_steps,
    'Hospital_expenditure': hospital_expenditure,
    'NUmber_of_past_hospitalizations': number_of_past_hospitalizations,
    'Anual_Salary': anual_salary,
    'region': region,
    'charges': charges
}

df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)

print(f'Successfully created sample_data.csv with {len(df)} rows.')