# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load data/model
df_segments = pd.read_csv("Grocery_Store_Customer_Segmentation_GUI/GUI/df_segments.csv")
rfm_data = pd.read_csv("Grocery_Store_Customer_Segmentation_GUI/GUI/rfm_segments.csv")
df1 = pd.read_csv('Grocery_Store_Customer_Segmentation_GUI/GUI/Products_with_Categories.csv')
df2 = pd.read_csv('Grocery_Store_Customer_Segmentation_GUI/GUI/Transactions.csv')
df = pd.merge(df2, df1, on='productId', how='inner')
df['purchase_amount'] = df['price'] * df['items']
product = df.groupby('productName').agg({'price':'mean', 'items':'sum', 'purchase_amount':'sum', 'Category': lambda x: x.mode().iloc[0]}).reset_index()

@st.cache_resource
def load_model():
    with open("Grocery_Store_Customer_Segmentation_GUI/GUI/kmeans_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Sidebar Navigation
section = st.sidebar.radio("Outline", options=["📚 Overview", "📊 Insights & Results", "👪 Customer Segmentation"])
st.sidebar.markdown("---")
# Thông tin nhóm thực hiện
st.sidebar.markdown("#### ✨ Thực hiện bởi:")
st.sidebar.markdown("""
                    - Nguyễn Nhật Tố Trân
                    - Nguyễn Vũ Mai Phương""")
# Thông tin giảng viên
st.sidebar.markdown("#### 👩‍🏫 Giảng viên: Cô Khuất Thùy Phương")
# Ngày báo cáo
st.sidebar.markdown("#### 📅 Thời gian: 04/2025")
st.sidebar.markdown("---")

# 1. Giới thiệu project
if section == "📚 Overview":
    st.title("👋 Welcome to Customer Segmentation App of a Grocery Store")
    st.image('Images/grocery_store.jfif', use_container_width =True)

    st.subheader("❓ Business")
    st.markdown("Đây là một cửa hàng tạp hóa bán sản phẩm thiết yếu như rau, củ, quả, thịt, cá, trứng, sữa, nước giải khát...  và khách hàng của họ là những người mua lẻ, với mong muốn có thể bán được nhiều hàng hóa hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.")

    st.subheader("🎯 Solution")
    st.markdown("""
    Dự án **Customer Segmentation** sử dụng phương pháp phân tích **RFM (Recency, Frequency, Monetary)** 
    nhằm chia khách hàng thành các nhóm để phục vụ các chiến lược marketing khác nhau.  
    - **Recency**: Số ngày kể từ lần mua hàng gần nhất  
    - **Frequency**: Số lần mua hàng  
    - **Monetary**: Tổng giá trị đơn hàng  

    Kết hợp phương pháp **RFM** và thuật toán **KMeans**, hệ thống phân nhóm khách hàng giúp tăng hiệu quả trong việc cá nhân hóa chăm sóc, giữ chân khách hàng, đồng thời giúp doanh nghiệp tăng tăng doanh thu.
    """)

# 2. Kết quả
elif section == "📊 Insights & Results":
    st.title("📊 Insights & Results")

    st.subheader("📄 Tổng quan về Dataset")
    num_rows = st.slider("Chọn số dòng hiển thị ngẫu nhiên", min_value=1, max_value=100, value=5, step=1)
    # Hiển thị dataframe với số dòng được chọn
    st.dataframe(df.head(num_rows))

    st.markdown(""" 
    - Các giao dịch được ghi nhận trong khoảng thời gian từ 01-01-2014 đến 30-12-2015
    - Trong 2 năm có 3,898 khách hàng thực hiện tổng cộng 77.000 lượt mua với tổng doanh thu là 331,000$
    - Cửa hàng bán tổng cộng 167 sản phẩm            
    """)

    st.subheader("🔎 Insights")
    st.markdown("#### 📌 Sản phẩm")

    # Vẽ biểu đồ Doanh thu theo sản phẩm
    top_n_revenue = st.slider("Chọn số lượng sản phẩm hiển thị (doanh thu)", min_value=2, max_value=len(product), value=5, step=1)
    top_df_revenue = product.sort_values(by='purchase_amount', ascending=False).head(top_n_revenue)
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    sns.barplot(x='productName', y='purchase_amount', data=top_df_revenue, palette='Blues_d', ax=ax1, ci=None)
    ax1.set_title("Doanh thu theo sản phẩm")
    ax1.set_xlabel("Sản phẩm")
    ax1.set_ylabel("Doanh thu")
    st.pyplot(fig1)
    st.markdown("- Sản phẩm mang lại doanh thu nhiều nhất : Thịt bò, trái cây nhiệt đới, khăn giấy, phô mai tươi, sô cô la đặc sản")

    # Vẽ biểu đồ Lượng sản phẩm tiêu thụ
    top_n_items = st.slider("Chọn số lượng sản phẩm hiển thị (số lượng)", min_value=2, max_value=len(product), value=5, step=1)
    top_df_items = product.sort_values(by='items', ascending=False).head(top_n_items)
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    sns.barplot(x='productName', y='items', data=top_df_items, palette='pink', ax=ax2, ci=None)
    ax2.set_title("Lượng sản phẩm tiêu thụ")
    ax2.set_xlabel("Sản phẩm")
    ax2.set_ylabel("Số lượng tiêu thụ")
    st.pyplot(fig2)
    st.markdown("""
    - Sản phẩm bán chạy nhất : Sữa tươi nguyên chất, rau củ khác, bánh mì cuộn, soda, sữa chua
    - Sản phẩm bán chậm nhất (mang lại ít doanh thu nhất) : Gà đông lạnh, cồn sát trùng, nước tẩy trang, sản phẩm bảo quản, dụng cụ nhà bếp
                    """)
    
    # Vẽ biểu đồ Giá sản phẩm
    top_n_price = st.slider("Chọn số lượng sản phẩm hiển thị (giá bán)", min_value=2, max_value=len(product), value=5, step=1)
    top_df_price = product.sort_values(by='price', ascending=False).head(top_n_price)
    fig3, ax3 = plt.subplots(figsize=(15, 8))
    sns.barplot(x='productName', y='price', data=top_df_price, palette='Greens', ax=ax3, ci=None)
    ax3.set_title("Giá bán theo sản phẩm")
    ax3.set_xlabel("Sản phẩm")
    ax3.set_ylabel("Giá bán")
    st.pyplot(fig3)
    st.markdown("""
    - Sản phẩm có giá cao nhất : Rượu Whisky, Mỹ phẩm cho bé, Khăn ăn, Rượu Prosecco, Túi xách
    - Sản phẩm có giá thấp nhất : Bánh mỳ, Gum, Nước đóng chai, Snack, Sản phẩm ăn liền
                    """)

    st.markdown("#### 📌 Ngành hàng")
    st.image('Images/Cate_Analysis.png', use_container_width =True)
    st.markdown("""
    - Thực phẩm tươi sống đóng góp 1/3 doanh thu
    - Ngàng hàng bán chạy nhất, đóng góp hơn 60% doanh thu : Thực phẩm tươi sống, Sản phẩm từ sữa, Bánh mì & đồ ngọt
    - Sản phẩm chăm sóc thú cưng, đồ ăn vặt, chăm sóc cá nhân là các ngành hàng bán chậm nhất của cửa hàng
                    """)

    st.markdown("#### 📌 Doanh thu và khách hàng")
    st.image('Images/Number of Sales Weekly.png', use_container_width =True)
    st.markdown("""
    - Biểu đồ **Tổng doanh thu theo tuần** có sự biến động rõ rệt, nguyên nhân có thể đến từ lượng khách hàng đến mua tại cửa hàng không ổn định. 
    - Nhìn chung doanh thu năm 2015 tăng hơn so với năm 2014.
                    """)
    
    st.image('Images/Number of Customers Weekly.png', use_container_width =True)
    st.image('Images/Sales per customer weekly.png', use_container_width =True)
    st.markdown("""
    - Biểu đồ **Khách hàng theo tuần** có sự tăng giảm liên tục. 
    - Lượng khách năm 2014 có xu hướng tăng nhẹ nhưng sang năm 2015 biểu đồ có xu hướng giảm, cửa hàng đã mất đi lượng khách.
    - Dù lượng khách hàng đến mua không ổn định, nhưng **Doanh thu trên từng khách hàng** có xu hướng tăng qua 2 năm. Lượng doanh thu tăng đột biến bắt đầu từ đầu năm 2015 và duy trì ổn định đến hết năm 2015. Có thể thấy rằng tuy lượng khách của cửa hàng không ổn định, thế nhưng cửa hàng vẫn giữ được một lượng khách trung thành đóng góp phần lớn doanh thu cho cửa hàng.
                    """)

    st.subheader("📈 Kết quả phân cụm")

    st.markdown(""" Thống kê theo **RFM** cho thấy:
    - Phần lớn khách hàng đã mua hàng 5 tháng gần đây, biểu đồ Recency lệch phải, đây là tín hiệu tốt cho cửa hàng khi thời gian mua hàng càng xa thì số lượng khách hàng giảm mạnh
    - Tập trung ở khoảng 2-6 lần mua hàng
    - Phần lớn khách hàng chi dưới 100$, và số lượng khách hàng chi nhiều hơn sẽ giảm dần.
                """) 
    st.image('Images/RFM.png', use_container_width =True)

    st.markdown("Phân chia khách hàng thành 5 nhóm dựa trên mô hình **KMeans**, các giá trị Recency, Frequency, Monetary trung bình, Tỷ lệ số lượng khách hàng của mỗi nhóm và doanh thu đóng góp tương ứng:")
    st.dataframe(rfm_data.head())
    st.image('Images/customer_segmentation.png', use_container_width =True)
    st.markdown("""
    - **Hardcore** : Chiếm 15% số lượng khách hàng của cửa hàng, nhưng mang lại doanh thu lớn nhất khi mua hàng thường xuyên và chi tiêu rất nhiều  
    - **Loyal** : Là nhóm khách hàng mua hàng thường xuyên, chiếm 31% số lượng  
    - **Potential** : Nhóm khách hàng tiềm năng, số lượng và chi tiêu xấp xỉ nhóm Loyal, tuy nhiên thời gian mua hàng lâu    
    - **At Risk** : Nhóm khách hàng có nguy cơ rời bỏ cửa hàng khi đã lâu không mua hàng (chiếm 18%)
    - **Lost** : Nhóm khách hàng không còn tương tác với cửa hàng (chỉ chiếm 8%) và giá trị mua hàng rất thấp
    """)

# 3. Tra cứu khách hàng
elif section == "👪 Customer Segmentation":
    st.title("👪 Tra cứu khách hàng")

    option = st.radio("Chọn phương thức tra cứu:", ["🔑 Nhập mã khách hàng", "✍️ Chọn giá trị RFM", "📁 Upload file"])

    if option == "🔑 Nhập mã khách hàng":
        customer_list = sorted(df_segments['Member_number'].unique())
        selected_customers = st.multiselect("Chọn một hoặc nhiều mã khách hàng:", customer_list)
        if selected_customers:
            result_segments = df_segments[df_segments["Member_number"].isin(selected_customers)]
            result_trans = df[df["Member_number"].isin(selected_customers)]
            if not result_segments.empty:
                st.success(f"✅ Tìm thấy {len(result_segments)} khách hàng:")
                st.dataframe(result_segments)
                st.success(f"✅ Lịch sử giao dịch của {len(result_segments)} khách hàng trên:")
                st.dataframe(result_trans)
            else:
               st.warning("⚠️ Không tìm thấy khách hàng nào.")

    elif option == "✍️ Chọn giá trị RFM":
        recency = st.slider("Recency (Số ngày)", min_value=1, max_value=600, value=30, step=1)
        frequency = st.slider("Frequency (Số lần mua hàng)", min_value=1, max_value=30, value=5, step=1)
        monetary = st.slider("Monetary ($)", min_value=1, max_value=600, value=50, step=5)
        cluster_names = {0: "Potential", 1: "Lost", 2: "Hardcore", 3: "Loyal", 4: "At Risk"}
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        cluster = model.predict(input_df)[0]
        segment_name = cluster_names.get(cluster, "Unknown")
        st.success(f"✅ Dự đoán khách hàng thuộc nhóm: **{segment_name}**")

    elif option == "📁 Upload file":
        uploaded_file = st.file_uploader("Upload file CSV chứa các thông tin Member_number, Recency, Frequency, Monetary", type="csv")
        if uploaded_file:
            df_uploaded = pd.read_csv(Grocery_Store_Customer_Segmentation_GUI/GUI/uploaded_file)
            df_uploaded["Cluster"] = model.predict(df_uploaded[['Recency', 'Frequency', 'Monetary']])
            cluster_names = {0: "Potential", 1: "Lost", 2: "Hardcore", 3: "Loyal", 4: "At Risk"}
            df_uploaded["Segment Name"] = df_uploaded["Cluster"].map(cluster_names)
            df_uploaded.drop('Cluster', axis=1, inplace=True)
            st.success("✅ Kết quả phân nhóm:")
            st.dataframe(df_uploaded)
            # Tìm lịch sử giao dịch
            matched_customers = df[df["Member_number"].isin(df_uploaded['Member_number'])]
            if not matched_customers.empty:
                st.success(f"✅ Lịch sử giao dịch của {matched_customers['Member_number'].nunique()} khách hàng trên:")
                st.dataframe(matched_customers)
            else:
                st.warning("⚠️ Không tìm thấy lịch sử giao dịch của các khách hàng trên.")
