#yêu cầu: nhập tên sản phẩm, đếm số pos, neg của sản phẩm
#database : 2 file csv
#giao diên: chọn file, nhập tên sp cần tra cứu, text nhập số sẳn phẩm có neg>=x%
#làm sao để từ tên sp ta lấy dc hết các cmt của sp đó
#từ bảng chứa tên sp lấy product id của sp, từ product id lấy hết tất cả các cmt có prodcut id đó , rồi cho cmt qua model
import  streamlit as st
import pandas as pd
import pickle
import  matplotlib.pyplot as plt
import numpy as np


st.set_page_config(layout="wide")

col1,col2=st.columns(2)

with col1:
    file_product_info = st.file_uploader("Choose a info file")
    if file_product_info is not None:
        dataframe_info = pd.read_csv(file_product_info)
        dataframe_info = dataframe_info[["product_id", "product_name"]]
        dataframe_info = dataframe_info[dataframe_info.isna().any(axis=1) == False]
        st.write(dataframe_info)


with col2:
    file_reviews=st.file_uploader("Choose a reviews file")
    if file_reviews is not None:
        dataframe_rv = pd.read_csv(file_reviews)
        dataframe_rv=dataframe_rv[["review_text", "is_recommended","product_id"]]
        dataframe_rv.rename(columns={'is_recommended': 'label', 'review_text': 'text'}, inplace=True)
        dataframe_rv = dataframe_rv[dataframe_rv.isna().any(axis=1) == False]
        st.write(dataframe_rv)


col3,col4=st.columns(2)

# nút tra cứu chỉ thực hiện khi có đầy đủ thông tin, nếu k thì ko thực hiện
with col3:
    name_product=st.text_input("name product","nhập vô đây")
    #st.write(name_product)
    button=st.button("tra cứu")
    if button and name_product and file_reviews is not None and file_product_info is not None:
        product_ids=dataframe_info.product_id[dataframe_info.product_name==name_product]
        #st.write(product_ids)
        #một tên sp có nhiều product id
        #xem xét tồn tại product id trong file đánh giá không
        lst_X=[]
        for product_id in product_ids:
            #st.write(product_id)
            if product_id not in dataframe_rv.product_id.unique():
                st.write("sản phẩm không có đánh giá nào")
            else:
                #lấy những hàng thỏa mã đk, trên cột text
                X_text=dataframe_rv.text[dataframe_rv.product_id==product_id]#phép toán trong ngoặc vuông trả về cột các gtri true false và chỉ lấy các hàng có gtri true
                lst_X.append(X_text)
                #st.write(X_text)
        X=pd.concat(lst_X,axis=0)
        st.write(X)
        with open('randomforest_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Sử dụng mô hình để dự đoán
        result = loaded_model.predict(X)
        cls,num=np.unique(result,return_counts=True)
        # st.write(result)
        # st.write(cls,num)# neg pos

        #https://matplotlib.org/3.5.3/tutorials/introductory/pyplot.html
        fig, ax = plt.subplots()
        names=["negative","positive"]
        posi=np.arange(len(names))
        ax = plt.xticks(posi, names)
        ax = plt.bar(posi, num, width=0.4, color=['cyan', 'skyblue'])  # vẽ biểu đồ cột
        ax = plt.title("Thống kê đánh giá")
        ax = plt.legend()
        st.pyplot(fig)





