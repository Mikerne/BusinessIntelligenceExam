{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b6062e34-e374-4bdc-bbcc-c84de6a5e68e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enhanced dataset saved as '../data/processed/heart_disease_clean_v2.csv'\n",
      "CHD incidence:\n",
      "TenYearCHD\n",
      "0    0.848113\n",
      "1    0.151887\n",
      "Name: proportion, dtype: float64\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "module 'pandas' has no attribute 'head'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[17], line 66\u001b[0m\n\u001b[1;32m     63\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCHD incidence:\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m     64\u001b[0m \u001b[38;5;28mprint\u001b[39m(df[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mTenYearCHD\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m.\u001b[39mvalue_counts(normalize\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m))\n\u001b[0;32m---> 66\u001b[0m pd\u001b[38;5;241m.\u001b[39mhead(\u001b[38;5;241m10\u001b[39m)\n",
      "\u001b[0;31mAttributeError\u001b[0m: module 'pandas' has no attribute 'head'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# ----------------------------\n",
    "# 1. Load Data\n",
    "# ----------------------------\n",
    "df = pd.read_csv(\"../data/raw/heart_disease.csv\")\n",
    "\n",
    "# ----------------------------\n",
    "# 2. Select Relevant Columns\n",
    "# ----------------------------\n",
    "relevant_cols = [\n",
    "    'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', \n",
    "    'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', \n",
    "    'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD'\n",
    "]\n",
    "df = df[relevant_cols]\n",
    "\n",
    "# ----------------------------\n",
    "# 3. Fill Missing Values\n",
    "# ----------------------------\n",
    "# Numerical\n",
    "num_cols = df.select_dtypes(include=np.number).columns.tolist()\n",
    "for col in num_cols:\n",
    "    df[col] = df[col].fillna(df[col].median())\n",
    "\n",
    "# Categorical\n",
    "cat_cols = ['education', 'BPMeds']\n",
    "for col in cat_cols:\n",
    "    df[col] = df[col].fillna(df[col].mode()[0])\n",
    "\n",
    "# ----------------------------\n",
    "# 4. Outlier Capping (1st/99th percentile)\n",
    "# ----------------------------\n",
    "for col in num_cols:\n",
    "    lower = df[col].quantile(0.01)\n",
    "    upper = df[col].quantile(0.99)\n",
    "    df[col] = np.clip(df[col], lower, upper)\n",
    "\n",
    "# ----------------------------\n",
    "# 5. Feature Engineering\n",
    "# ----------------------------\n",
    "# Pulse pressure\n",
    "df['pulse_pressure'] = df['sysBP'] - df['diaBP']\n",
    "\n",
    "# Approximate smoking exposure (pack-years)\n",
    "df['smoking_pack_years'] = df['cigsPerDay'] * (df['age'] - 18).clip(lower=0)\n",
    "\n",
    "# ----------------------------\n",
    "# 6. Ensure No Missing Values\n",
    "# ----------------------------\n",
    "assert df.isna().sum().sum() == 0, \"Data still contains missing values!\"\n",
    "\n",
    "# ----------------------------\n",
    "# 7. Save Enhanced Dataset\n",
    "# ----------------------------\n",
    "df.to_csv(\"../data/processed/heart_disease_clean_v2.csv\", index=False)\n",
    "print(\"Enhanced dataset saved as '../data/processed/heart_disease_clean_v2.csv'\")\n",
    "\n",
    "# ----------------------------\n",
    "# 8. Optional: Class Distribution\n",
    "# ----------------------------\n",
    "print(\"CHD incidence:\")\n",
    "print(df['TenYearCHD'].value_counts(normalize=True))\n",
    "\n",
    "df.head(10)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9181731-3901-428a-8473-6f636e529347",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:anaconda3]",
   "language": "python",
   "name": "conda-env-anaconda3-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
