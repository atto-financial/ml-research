import pandas as pd
import numpy as np
import mysql.connector
import joblib

# เชื่อมต่อ MySQL
conn = mysql.connector.connect(
    host="switchyard.proxy.rlwy.net",
    user="root",
    password="TjndDsrCcjBuxfgQlvhzNnoVCXOapmUG",
    database="railway",
    port=16533
)
cursor = conn.cursor()

# โหลดโมเดลจากไฟล์ .pkl
model = joblib.load('mlrfth50_v1.0.pkl')
print("load model.pkl complete")

# Test
# ดึงข้อมูลลูกค้ารายใหม่จากตาราง new_customers
# query_new = """
# SELECT set_1 AS set1, set_2 AS set2, set_3 AS set3, set_4 AS set4, 
#        set_5 AS set5, set_6 AS set6, set_7 AS set7, set_8 AS set8, 
#        set_9 AS set9, set_10 AS set10, set_11 AS set11,
#        customer_id
# FROM new_customers
# WHERE processed = 0
# """

set_ans = [2, 2, 2, 2, 3, 3, 1, 1, 1, 2, 2] 

set_df = pd.DataFrame([set_ans], columns=[f'set{i+1}' for i in range(len(set_ans))])
print("set_df:", set_df)

print("X:", new_customers)

predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)[:, 1]

results = pd.DataFrame({
        'predicted_ust': predictions,
        'probability_ust': probabilities
    })

print("Predicted Result:")
print(results)


print("P-new customers ust = 1):", probabilities[0])
print("loan assesment (0 = approve loan, 1 = deny loan):", predictions[0])

# ***run model ใหม่แล้ว ปรับ threshold = 0.50 ปรับชื่อ model label เป็น  "mlrfth50_v1.0"***
# threshold = 0.50 
# predictions_adjusted = (probabilities >= threshold).astype(int)
# print(f"loan threshold adjusted = {threshold} :", predictions_adjusted[0])

new_customers = pd.read_sql(set_df, engine)

# ตรวจสอบว่ามีข้อมูลลูกค้ารายใหม่หรือไม่
if new_customers.empty:
    print("ไม่มีลูกค้ารายใหม่ที่รอประเมิน")
else:
    print("ลูกค้ารายใหม่ที่รอประเมิน:")
    print(new_customers.head())

    # เตรียมข้อมูลสำหรับทำนาย
    X_new = new_customers.drop("customer_id", axis=1)
    
    # ทำนาย
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]

    # รวมผลลัพธ์กับ customer_id
    results = pd.DataFrame({
        'customer_id': new_customers['customer_id'],
        'predicted_ust': predictions,
        'probability_ust1': probabilities
    })
    print("ผลการทำนาย:")
    print(results)

    # บันทึกผลลัพธ์ลงตาราง predictions
    results.to_sql('predictions', engine, if_exists='append', index=False)
    print("บันทึกผลการทำนายลงตาราง predictions เรียบร้อย")

    # อัปเดตสถานะ processed ใน new_customers
    update_query = """
    UPDATE new_customers 
    SET processed = 1
    WHERE customer_id IN (%s)
    """ % ','.join([str(cid) for cid in new_customers['customer_id']])
    cursor.execute(update_query)
    conn.commit()
    print("อัปเดตสถานะ processed ใน new_customers เรียบร้อย")

# ปิดการเชื่อมต่อ
cursor.close()
conn.close()
