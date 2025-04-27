import pandas as pd
import numpy as np

def preprocess_data(raw_dat):
    df = pd.DataFrame({
    'Q1': Q1, 'Q2': Q2, 'Q3': Q3, 'Q4': Q4, 'Q5': Q5, 'Q6': Q6, 'Q7': Q7, 'Q8': Q8
})
    
Q1 # Spending: จัดลำดับความสำคัญ
Q2 # Spending: ติดตามค่าใช้จ่าย
Q3 # Saving: พยายามเก็บเงิน
Q4 # Saving: ลดค่าใช้จ่ายเพื่อเก็บเงิน
Q5 # Borrowing: วางแผนชำระหนี้
Q6 # Borrowing: ชำระหนี้เก่า
Q7 # Planning: วางแผนการใช้เงิน
Q8 # Planning: หารายได้เสริม


df['Spending_Score'] = (df['Q1'] + df['Q2']) / 2  # คะแนนการใช้จ่าย
df['Saving_Score'] = (df['Q3'] + df['Q4']) / 2    # คะแนนการออม
df['Borrowing_Score'] = (df['Q5'] + df['Q6']) / 2 # คะแนนการยืม
df['Planning_Score'] = (df['Q7'] + df['Q8']) / 2  # คะแนนการวางแผน

df['Saving_to_Spending_Ratio'] = df['Saving_Score'] / df['Spending_Score']  # อัตราส่วนการออมต่อการใช้จ่าย

# สร้าง Target (สมมติว่าคะแนนรวมสูง = ชำระหนี้ได้)
df['Total_Score'] = df[['Spending_Score', 'Saving_Score', 'Borrowing_Score', 'Planning_Score']].sum(axis=1)
df['Target'] = np.where(df['Total_Score'] > df['Total_Score'].median(), 1, 0)


print(df.head())