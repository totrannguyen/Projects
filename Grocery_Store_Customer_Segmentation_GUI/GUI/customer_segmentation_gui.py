# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

# Load data/model & preprocessing
df_segments = pd.read_csv("df_segments.csv")
rfm_segments = pd.read_csv("rfm_segments.csv")
df1 = pd.read_csv('Products_with_Categories.csv')
df2 = pd.read_csv('Transactions.csv')
df = pd.merge(df2, df1, on='productId', how='inner')
df['purchase_amount'] = df['price'] * df['items']
df['Date'] = df['Date'].apply(lambda x : datetime.strptime(x, '%d-%m-%Y').date()).astype('datetime64[ns]')
df.drop_duplicates(inplace=True)
product = df.groupby('productName').agg({'price':'mean', 'items':'sum', 'purchase_amount':'sum', 'Category': lambda x: x.mode().iloc[0]}).reset_index()
category = df[['Category','items','purchase_amount']].groupby('Category').sum().sort_values(by='purchase_amount', ascending=False).reset_index()
category['percent_amount'] = category['purchase_amount']/sum(category['purchase_amount'])*100

@st.cache_resource
def load_model():
    with open("kmeans_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Hàm vẽ biểu đồ
def draw_barplot(data, x, y, title, xlabel, ylabel, palette, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x=x, y=y, palette=palette, ax=ax, ci=None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

def draw_pieplot(data, labels, autopct, colors, pctdistance, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    plt.pie(data, labels=labels, autopct=autopct, colors=colors, pctdistance=pctdistance)
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)

def draw_lineplot(x, y, labels, title_text):
    fig = px.line(x=x, y=y, labels=labels)
    fig.update_layout(title_text=title_text, title_x=0.4, title_font=dict(size=18))
    st.plotly_chart(fig, use_container_width=True)

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
st.sidebar.image('Images/logo.jpg', use_container_width =True)

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
    st.markdown("""<style>/* Tab đang được chọn - làm nổi bật */div[data-testid="stTabs"] button[aria-selected="true"] {color: #FF4B4B !important; border-bottom: 3px solid #FF4B4B !important; font-weight: 800 !important;}
                        /* Căn giữa tất cả các tab */div[data-testid="stTabs"] div[role="tablist"] {justify-content: center !important;}
                   </style>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔎 DATASET & INSIGHTS", "📈 KẾT QUẢ PHÂN CỤM", "⭐ ĐỀ XUẤT GIẢI PHÁP"])

    with tab1:
        st.subheader("📄 Tổng quan về Dataset")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            #st.metric("Số lượng khách hàng", df['Member_number'].nunique())
            st.markdown(f"""<div style="background-color: #2c2f33; padding: 8px; border-radius: 8px; text-align: center; color: white; line-height: 1.4;">
                                <div style="font-size: 14px; font-weight: 100;">Số lượng khách hàng</div>
                                <div style="font-size: 30px; color: lightgreen; font-weight: bold;">{df['Member_number'].nunique()}</div>
                            </div>""", unsafe_allow_html=True)
        with col2:
            #st.metric("Số lượng sản phẩm", df['productId'].nunique())
            st.markdown(f"""<div style="background-color: #2c2f33; padding: 8px; border-radius: 8px; text-align: center; color: white; line-height: 1.4;">
                                <div style="font-size: 14px; font-weight: 100;">Số lượng sản phẩm</div>
                                <div style="font-size: 30px; color: lightgreen; font-weight: bold;">{df['productId'].nunique()}</div>
                            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div style="background-color: #2c2f33; padding: 8px; border-radius: 8px; text-align: center; color: white; line-height: 1.4;">
                                <div style="font-size: 14px; font-weight: 100;">Lượng hàng đã bán</div>
                                <div style="font-size: 30px; color: lightgreen; font-weight: bold;">{df['items'].sum()}</div>
                            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div style="background-color: #2c2f33; padding: 8px; border-radius: 8px; text-align: center; color: white; line-height: 1.4;">
                                <div style="font-size: 14px; font-weight: 100;">Doanh thu ($)</div>
                                <div style="font-size: 30px; color: lightgreen; font-weight: bold;">{f"{round(sum(df['purchase_amount'])/1000000, 2)} M"}</div>
                            </div>""", unsafe_allow_html=True)
        st.write("")
        num_rows = st.slider("Chọn số dòng data hiển thị", min_value=1, max_value=100, value=5, step=1)
        # Hiển thị dataframe với số dòng được chọn
        st.dataframe(df.head(num_rows))

        st.markdown("- Các giao dịch được ghi nhận trong khoảng thời gian từ 01-01-2014 đến 30-12-2015")

        st.subheader("🔎 Insights")
        st.markdown("#### 📌 Sản phẩm")

        # Vẽ biểu đồ Doanh thu theo sản phẩm
        top_n_revenue = st.slider("Chọn số lượng sản phẩm hiển thị (doanh thu)", min_value=2, max_value=len(product), value=5, step=1)
        top_df_revenue = product.sort_values(by='purchase_amount', ascending=False).head(top_n_revenue)
        draw_barplot(top_df_revenue, 'productName', 'purchase_amount', "Doanh thu theo sản phẩm", "Sản phẩm", "Doanh thu", 'Blues_d', (15, 8))
        st.markdown("- Sản phẩm mang lại doanh thu nhiều nhất : Thịt bò, trái cây nhiệt đới, khăn giấy, phô mai tươi, sô cô la đặc sản")

        # Vẽ biểu đồ Lượng sản phẩm tiêu thụ
        top_n_items = st.slider("Chọn số lượng sản phẩm hiển thị (số lượng)", min_value=2, max_value=len(product), value=5, step=1)
        top_df_items = product.sort_values(by='items', ascending=False).head(top_n_items)
        draw_barplot(top_df_items, 'productName', 'items', "Lượng sản phẩm tiêu thụ", "Sản phẩm", "Số lượng tiêu thụ", 'pink', (15, 8))
        st.markdown("""
        - Sản phẩm bán chạy nhất : Sữa tươi nguyên chất, rau củ khác, bánh mì cuộn, soda, sữa chua
        - Sản phẩm bán chậm nhất (mang lại ít doanh thu nhất) : Gà đông lạnh, cồn sát trùng, nước tẩy trang, sản phẩm bảo quản, dụng cụ nhà bếp
                        """)
        
        # Vẽ biểu đồ Giá sản phẩm
        top_n_price = st.slider("Chọn số lượng sản phẩm hiển thị (giá bán)", min_value=2, max_value=len(product), value=5, step=1)
        top_df_price = product.sort_values(by='price', ascending=False).head(top_n_price)
        draw_barplot(top_df_price, 'productName', 'price', "Giá bán theo sản phẩm", "Sản phẩm", "Giá bán", 'Greens', (15, 8))
        st.markdown("""
        - Sản phẩm có giá cao nhất : Rượu Whisky, Mỹ phẩm cho bé, Khăn ăn, Rượu Prosecco, Túi xách
        - Sản phẩm có giá thấp nhất : Bánh mỳ, Gum, Nước đóng chai, Snack, Sản phẩm ăn liền
                        """)

        st.markdown("#### 📌 Ngành hàng")
        #st.image('Images/Cate_Analysis.png', use_container_width =True)
        draw_barplot(category, 'Category', 'purchase_amount', "Doanh thu theo ngành hàng", "Ngành hàng", "Doanh thu", 'rocket', (20,8))
        draw_barplot(category.sort_values(by='items', ascending=False), 'Category', 'items', "Lượng sản phẩm đã bán theo ngành hàng", "Ngành hàng", "Số lượng sản phẩm đã bán", 'YlOrBr', (20,8))
        draw_pieplot(category['purchase_amount'], category['Category'], '%1.1f%%', sns.color_palette("Set3", len(category)), 0.8, 'Tỷ lệ doanh thu theo ngành hàng', (8,8))
        st.markdown("""
        - Thực phẩm tươi sống đóng góp 1/3 doanh thu
        - Ngàng hàng bán chạy nhất, đóng góp hơn 60% doanh thu : Thực phẩm tươi sống, Sản phẩm từ sữa, Bánh mì & đồ ngọt
        - Sản phẩm chăm sóc thú cưng, đồ ăn vặt, chăm sóc cá nhân là các ngành hàng bán chậm nhất của cửa hàng
                        """)

        st.markdown("#### 📌 Doanh thu và khách hàng")
        sales_weekly = df.resample('W', on='Date').size()
        draw_lineplot(sales_weekly.index, sales_weekly.values, {'y': 'Number of Sales','x': 'Date'}, 'Number of Sales Weekly')
        st.markdown("""
        - Biểu đồ **Tổng doanh thu theo tuần** có sự biến động rõ rệt, nguyên nhân có thể đến từ lượng khách hàng đến mua tại cửa hàng không ổn định. 
        - Nhìn chung doanh thu năm 2015 tăng hơn so với năm 2014.
                        """)
        unique_customer_weekly = df.resample('w', on='Date').Member_number.nunique()
        draw_lineplot(unique_customer_weekly.index, unique_customer_weekly.values, {'y': 'Number of Customers','x': 'Number of Customers Weekly'}, 'Number of Customers Weekly')
        st.markdown("""
        - Biểu đồ **Khách hàng theo tuần** có sự tăng giảm liên tục. 
        - Lượng khách năm 2014 có xu hướng tăng nhẹ nhưng sang năm 2015 biểu đồ có xu hướng giảm, cửa hàng đã mất đi lượng khách.""")        
        sales_per_customer = sales_weekly / unique_customer_weekly
        draw_lineplot(sales_per_customer.index, sales_per_customer.values, {'y': 'Sales per Customer Ratio','x':'Date'}, 'Sales per Customer Weekly')
        st.markdown("- Dù lượng khách hàng đến mua không ổn định, nhưng **Doanh thu trên từng khách hàng** có xu hướng tăng qua 2 năm. Lượng doanh thu tăng đột biến bắt đầu từ đầu năm 2015 và duy trì ổn định đến hết năm 2015. Có thể thấy rằng tuy lượng khách của cửa hàng không ổn định, thế nhưng cửa hàng vẫn giữ được một lượng khách trung thành đóng góp phần lớn doanh thu cho cửa hàng.")
        
    with tab2:
        st.subheader("📌 Phân tích RFM")
        st.image('Images/RFM.png', use_container_width =True)
        st.markdown("""
        - Phần lớn khách hàng đã mua hàng 5 tháng gần đây, biểu đồ Recency lệch phải, đây là tín hiệu tốt cho cửa hàng khi thời gian mua hàng càng xa thì số lượng khách hàng giảm mạnh
        - Tập trung ở khoảng 2-6 lần mua hàng
        - Phần lớn khách hàng chi dưới 100$, và số lượng khách hàng chi nhiều hơn sẽ giảm dần.""") 
        
        st.subheader("📌 Xây dựng mô hình")
        st.markdown("""
                    - Tiến hành xây dựng mô hình phân cụm khách hàng bằng các thuật toán: **Manual Segmentation, KMeans, GMM, DBSCAN**, và **Hierarchical Clustering**, áp dụng lần lượt trên ba trường hợp dữ liệu: gốc, đã loại bỏ outliers, đã được chuẩn hóa (scaled) và xử lý trên môi trường PySpark.
                    - Mục tiêu là tìm được mô hình có sự cân bằng tối ưu giữa số lượng cụm khách hàng và chỉ số **Silhouette** nhằm đảm bảo tính phân biệt và chất lượng của các cụm""")
        phuong_phap = ['Manual', 'GMM(S)', 'GMM(RO)', 'GMM', 'DBSCAN', 'Hi(RO)', 'Hi(S)','DBSCAN(RO)', 'Hi', 'K(S)', 'K(5)', 'K(RO)', 'K(PS+RO)', 'DBSCAN(S)', 'K(PS+S)', 'K(2)', 'K(PS)']
        so_cum = [6, 6, 4 , 6, 2, 6, 4, 1, 6, 3, 5, 4, 6, 1, 5, 2, 5]
        silhouette_score = [-0.059, 0.13, 0.2, 0.2, 0.23, 0.23, 0.25, 0.3, 0.33, 0.36, 0.4, 0.42, 0.45, 0.45, 0.48, 0.57, 0.58]

        # Tạo biểu đồ
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Bar graph cho số cụm
        ax1.bar(phuong_phap, so_cum, color='skyblue', label='Số cụm')
        ax1.set_ylabel('Số cụm', color='blue')
        ax1.set_xlabel('Phương pháp')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, max(so_cum)+1)
        ax1.set_xticklabels(phuong_phap, rotation=45, ha='right')
        # Line graph cho Silhouette Score
        ax2 = ax1.twinx()
        ax2.plot(phuong_phap, silhouette_score, color='red', marker='o', label='Silhouette Score')
        ax2.set_ylabel('Silhouette Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(min(silhouette_score)-0.1, max(silhouette_score)+0.1)
        ax2.set_xticklabels(phuong_phap, rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("""
                    Chú thích biểu đồ:
                    - S : Scale, là trường hợp đã chuẩn hóa dữ liệu
                    - RO : Remove outliers, là trường hợp loại bỏ outliers
                    - PS : PySpark, được thực hiện trên môi trường PySpark
                    - Hi : thuật toán Hierarchical Clustering
                    - K : thuật toán KMeans 
                    """)
        st.markdown("Kết quả cho thấy, thuật toán **KMeans** vượt trội hơn so với các phương pháp còn lại ở cả ba trường hợp, với **Silhouette Score** cao và số lượng cụm hợp lý, tách biệt nhau, cho thấy khả năng phân tách và gom nhóm khách hàng hiệu quả nhất.")

        st.subheader("📌 Kết quả phân cụm khách hàng")
        st.markdown("Phân chia khách hàng thành 5 nhóm dựa trên mô hình **KMeans**, các giá trị Recency, Frequency, Monetary trung bình, Tỷ lệ số lượng khách hàng của mỗi nhóm và Doanh thu đóng góp tương ứng:")
        st.dataframe(rfm_segments.rename(columns={'Cluster':'Nhóm', 'Percent_Quantity':'Tỷ lệ', 'Percent_Revenue':'Doanh thu'}).head())
        st.image('Images/customer_segmentation.png', use_container_width =True)
        st.markdown("""
        - **Hardcore** : Chiếm 15% số lượng khách hàng của cửa hàng, nhưng mang lại doanh thu lớn nhất khi mua hàng thường xuyên và chi tiêu rất nhiều  
        - **Loyal** : Là nhóm khách hàng mua hàng thường xuyên, chiếm 31% số lượng  
        - **Potential** : Nhóm khách hàng tiềm năng, số lượng và chi tiêu xấp xỉ nhóm Loyal, tuy nhiên thời gian mua hàng lâu    
        - **At Risk** : Nhóm khách hàng có nguy cơ rời bỏ cửa hàng khi đã lâu không mua hàng (chiếm 18%)
        - **Lost** : Nhóm khách hàng không còn tương tác với cửa hàng (chỉ chiếm 8%) và giá trị mua hàng rất thấp
        """)

    with tab3:
        st.subheader("🔵 Biểu đồ Doanh thu theo từng nhóm khách hàng")
        st.markdown('') 
        draw_pieplot(rfm_segments['Percent_Revenue'], rfm_segments['Cluster'], '%1.1f%%', sns.color_palette("Pastel1", len(rfm_segments)), 0.8, '', (4,4))
        st.markdown('')   
        st.subheader("🛍️ Đề xuất chiến lược tiếp cận từng nhóm khách hàng")
        st.markdown('') 
        segments = [{"Segment": "Hardcore","Tỷ lệ": "15.4%","Đặc điểm": "R: Cao | F: Cao | M: Cao","Doanh thu": "32%","Mục tiêu": "Giữ chân, tăng giá trị đơn","Phương pháp": "Chương trình khách hàng thân thiết cao cấp, dịch vụ cá nhân hóa, cross-sell"},
                    {"Segment": "Loyal","Tỷ lệ": "31%","Đặc điểm": "R: Cao | F: Cao | M: TB","Doanh thu": "26%","Mục tiêu": "Tăng giá trị đơn","Phương pháp": "Tích điểm thành viên, khuyến mãi combo, Ưu đãi sinh nhật & dịp đặc biệt"},
                    {"Segment": "Potential","Tỷ lệ": "27.6%","Đặc điểm": "R: TB | F: Cao | M: TB","Doanh thu": "25.5%","Mục tiêu": "Tăng tần suất mua","Phương pháp": "Ưu đãi cá nhân hóa diễn ra trong thời gian ngắn, gửi tin SMS/mail"},
                    {"Segment": "At Risk","Tỷ lệ": "18.3%","Đặc điểm": "R: Thấp | F: TB | M: Thấp","Doanh thu": "13.3%","Mục tiêu": "Lôi kéo trở lại","Phương pháp": "Khảo sát & cải thiện dịch vụ, giảm giá đặc biệt hoặc quà tặng"},
                    {"Segment": "Lost","Tỷ lệ": "7.7%","Đặc điểm": "R: Thấp | F: Thấp | M: Thấp","Doanh thu": "3.4%","Mục tiêu": "Cân nhắc nguồn lực, có thể bỏ qua nhóm này","Phương pháp": "Khảo sát lý do rời bỏ, nhóm chiếm tỷ lệ ít nên có thể bỏ qua"},]
        for seg in segments:
            with st.container():
                st.markdown(f"""<div style='border:1px solid #ccc; border-radius:10px; padding:15px; margin-bottom:10px; background-color:#000000'>
                <h4 style='margin-bottom:5px'>📌 <b>{seg['Segment']}</b> | Tỷ lệ: {seg['Tỷ lệ']} | Doanh thu: {seg['Doanh thu']}</h4>
                <p><b>🎯 Mục tiêu:</b> {seg['Mục tiêu']}</p>
                <p><b>🔍 Đặc điểm RFM:</b> {seg['Đặc điểm']}</p>
                <p><b>🛠️ Đề xuất tiếp cận:</b> {seg['Phương pháp']}</p>
                </div>""", unsafe_allow_html=True)
        st.image('Images/shopping.jfif', use_container_width =True)

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
                result_trans=result_trans.merge(result_segments[['Member_number', 'Segment']], on='Member_number', how='left')
                result_trans=result_trans.rename(columns={'Member_number':'Mã khách hàng',	'Date':'Thời gian',	'productId':'Mã sản phẩm','items':'Số lượng',	'productName':'Sản phẩm','price':'Đơn giá','Category':'Ngành hàng','purchase_amount':'Tổng tiền','Segment':'Nhóm'})
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
            df_uploaded = pd.read_csv(uploaded_file)
            df_uploaded["Cluster"] = model.predict(df_uploaded[['Recency', 'Frequency', 'Monetary']])
            cluster_names = {0: "Potential", 1: "Lost", 2: "Hardcore", 3: "Loyal", 4: "At Risk"}
            df_uploaded["Segment"] = df_uploaded["Cluster"].map(cluster_names)
            df_uploaded.drop('Cluster', axis=1, inplace=True)
            st.success("✅ Kết quả phân nhóm:")
            st.dataframe(df_uploaded)
            # Tìm lịch sử giao dịch
            matched_customers = df[df["Member_number"].isin(df_uploaded['Member_number'])]
            if not matched_customers.empty:
                st.success(f"✅ Lịch sử giao dịch của {matched_customers['Member_number'].nunique()} khách hàng trên:")
                matched_customers=matched_customers.merge(df_uploaded[['Member_number', 'Segment']], on='Member_number', how='left')
                matched_customers=matched_customers.rename(columns={'Member_number':'Mã khách hàng',	'Date':'Thời gian',	'productId':'Mã sản phẩm','items':'Số lượng',	'productName':'Sản phẩm','price':'Đơn giá','Category':'Ngành hàng','purchase_amount':'Tổng tiền','Segment':'Nhóm'})
                st.dataframe(matched_customers)
            else:
                st.warning("⚠️ Không tìm thấy lịch sử giao dịch của các khách hàng trên.")
