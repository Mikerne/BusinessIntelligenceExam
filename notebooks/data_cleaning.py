{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b6062e34-e374-4bdc-bbcc-c84de6a5e68e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Renset datasæt gemt som '../data/processed/heart_disease_clean.csv'\n"
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
    "# 3. Clean Data\n",
    "# ----------------------------\n",
    "# Fill missing numerical values with median\n",
    "num_cols = df.select_dtypes(include=np.number).columns.tolist()\n",
    "for col in num_cols:\n",
    "    df[col] = df[col].fillna(df[col].median())\n",
    "\n",
    "# Fill missing categorical values with mode\n",
    "cat_cols = ['education', 'BPMeds']\n",
    "for col in cat_cols:\n",
    "    df[col] = df[col].fillna(df[col].mode()[0])\n",
    "\n",
    "# Ensure no missing values remain\n",
    "assert df.isna().sum().sum() == 0, \"Data still contains missing values!\"\n",
    "\n",
    "# ----------------------------\n",
    "# 4. Save Cleaned Dataset\n",
    "# ----------------------------\n",
    "df.to_csv(\"../data/processed/heart_disease_clean.csv\", index=False)\n",
    "print(\"Renset datasæt gemt som '../data/processed/heart_disease_clean.csv'\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2722e1d-1a57-4768-99af-2127c0b97f00",
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
